import assert from "node:assert/strict";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test, type TestContext } from "node:test";
import {
  createAssistantMessageEventStream,
  getApiProvider,
  registerApiProvider,
  unregisterApiProviders,
  type Api,
  type AssistantMessage,
  type Context,
  type Message,
  type Model,
  type ProviderStreamOptions,
  type ToolCall,
} from "@earendil-works/pi-ai";
import { streamOpenAICompletions } from "@earendil-works/pi-ai/openai-completions";
import type { ExtensionContext } from "@earendil-works/pi-coding-agent";
import { createExpertRegistry } from "../tools/expert-registry.js";
import { restoreExpertSessions, runExpertTurn, type ExpertTurnInput } from "../tools/expert-turn.js";
import { createSourceContext } from "../tools/source-context.js";
import { loadExpertTasks } from "../utils/expert-store.js";

const model: Model<"openai-completions"> = {
  api: "openai-completions", provider: "test", id: "expert", name: "Test expert",
  baseUrl: "http://127.0.0.1:1/v1", reasoning: false, input: ["text"],
  contextWindow: 8192, maxTokens: 1024,
  cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
};

function reply(content: AssistantMessage["content"], stopReason: AssistantMessage["stopReason"] = "stop", cost = 0): AssistantMessage {
  return {
    role: "assistant", api: model.api, provider: model.provider, model: model.id,
    content, stopReason, timestamp: 1,
    usage: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0,
      cost: { input: 0, output: cost, cacheRead: 0, cacheWrite: 0, total: cost } },
  };
}

const answer = (text = "DONE") => reply([{ type: "text", text }]);
const call = (id: number): ToolCall => ({ type: "toolCall", id: `call-${id}`, name: "read_file", arguments: { path: "fixture.txt" } });
const calls = (count: number, start = 0) => reply(Array.from({ length: count }, (_, i) => call(start + i)), "toolUse");

type Request = { context: Context; options?: ProviderStreamOptions };
type ScriptedReply = AssistantMessage | ((request: Request) => AssistantMessage);

function fixture(t: TestContext, script: ScriptedReply[], api: Api = model.api) {
  const cwd = mkdtempSync(join(tmpdir(), "chronos-expert-test-"));
  t.after(() => rmSync(cwd, { recursive: true, force: true }));
  writeFileSync(join(cwd, "fixture.txt"), "fixture contents");
  const requests: Request[] = [];
  const testModel = { ...model, api };
  const previous = getApiProvider(api);
  const stream = (_model: Model<Api>, context: Context, options?: ProviderStreamOptions) => {
    assert.ok(requests.length < 12, "Expert loop did not terminate");
    const request = { context: structuredClone(context), options };
    requests.push(request);
    const next = script.shift();
    assert.ok(next, "Unexpected completion request");
    const result = typeof next === "function" ? next(request) : next;
    const events = createAssistantMessageEventStream();
    events.end(result);
    return events;
  };
  registerApiProvider({
    api, stream,
    streamSimple: (model, context, options) => stream(model, context, { ...options }),
  }, "chronos-tests");
  t.after(() => {
    unregisterApiProviders("chronos-tests");
    if (previous) registerApiProvider(previous);
  });
  const extCtx = {
    cwd, model: testModel,
    modelRegistry: { find: () => testModel, getApiKey: async () => "test-key" },
    sessionManager: { getSessionId: () => "test-session" },
  } as unknown as ExtensionContext;
  const registry = createExpertRegistry();
  const sourceCtx = createSourceContext();
  return {
    cwd, extCtx, registry, sourceCtx, requests,
    run: (input: Partial<ExpertTurnInput> = {}) => runExpertTurn(registry, sourceCtx, "Expert instructions.", extCtx, { prompt: "Inspect the fixture.", ...input }),
    stored: () => loadExpertTasks(cwd, "test-session"),
    history: () => registry.sessions.get("task-1")!.messages,
  };
}

function assertPaired(messages: Message[]) {
  const pending = new Set<string>();
  for (const message of messages) {
    if (message.role === "assistant") {
      assert.equal(pending.size, 0, "Assistant replied before all tool results");
      for (const block of message.content) if (block.type === "toolCall") pending.add(block.id);
    } else if (message.role === "toolResult") {
      assert.ok(pending.delete(message.toolCallId), "Unexpected or duplicate tool result");
    } else {
      assert.equal(pending.size, 0, "User message interrupted tool results");
    }
  }
  assert.equal(pending.size, 0, "Unanswered tool calls");
}

test("normal answers and under-budget tools preserve the conversation", async (t) => {
  const f = fixture(t, [calls(1), answer(), answer("FOLLOW-UP")]);
  const result = await f.run();
  assert.ok(result.ok);
  assert.equal(result.text, "DONE");
  assert.equal(result.toolUses.length, 1);
  const followUp = await f.run({ taskId: result.taskId, prompt: "Continue." });
  assert.ok(followUp.ok);
  assert.equal(followUp.text, "FOLLOW-UP");
  assert.equal(f.stored()[0].turns.length, 2);
  assertPaired(f.history());
  assert.ok(f.requests.every((r) => r.options?.toolChoice === undefined));
});

test("budget exhaustion retains definitions and serializes tools with tool_choice none", async (t) => {
  const f = fixture(t, [...Array.from({ length: 8 }, (_, i) => calls(1, i)), answer()]);
  const result = await f.run();
  assert.ok(result.ok);
  assert.equal(result.text, "DONE");
  assert.equal(result.toolUses.length, 8);
  for (const request of f.requests) assert.deepEqual(request.context.tools, f.requests[0].context.tools);
  const last = f.requests.at(-1)!;
  assert.equal(last.options?.toolChoice, "none");
  assert.match(last.context.systemPrompt!, /budget.*exhausted/i);
  assertPaired(last.context.messages);

  // Inspect the real, unpatched pi-ai serializer without making a network call.
  let payload: Record<string, unknown> | undefined;
  const serialized = await streamOpenAICompletions(model, last.context, {
    apiKey: "test-key", toolChoice: "none",
    onPayload(value) {
      payload = value as Record<string, unknown>;
      throw new Error("Captured before network");
    },
  }).result();
  assert.match(serialized.errorMessage!, /Captured before network/);
  assert.ok(Array.isArray(payload?.tools) && payload.tools.length > 0);
  assert.equal(payload.tool_choice, "none");
});

test("oversized batches execute only eight calls and pair rejected calls", async (t) => {
  const overflow = calls(10);
  // A real execution would overwrite the fixture, making the cap observable.
  overflow.content[8] = { type: "toolCall", id: "call-8", name: "write_file", arguments: { path: "fixture.txt", content: "OVERWRITTEN" } };
  const f = fixture(t, [overflow, answer()]);
  const result = await f.run({ grantedCaps: ["write"] });
  assert.ok(result.ok);
  assert.equal(readFileSync(join(f.cwd, "fixture.txt"), "utf8"), "fixture contents");
  assert.equal(result.toolUses.filter((use) => !use.isError).length, 8);
  assert.equal(result.toolUses.filter((use) => use.isError).length, 2);
  assertPaired(f.history());
});

test("post-budget calls with unexpected stop reasons recover and survive restore", async (t) => {
  const f = fixture(t, [calls(8), reply([call(8)], "stop"), reply([call(9)], "length"), answer(), answer("RESTORED")]);
  const result = await f.run();
  assert.ok(result.ok);
  assert.equal(result.text, "DONE");
  assert.deepEqual(result.toolUses.slice(8).map((use) => use.isError), [true, true]);
  for (const request of f.requests) assertPaired(request.context.messages);
  const restored = createExpertRegistry();
  await restoreExpertSessions(restored, f.extCtx, f.stored());
  // Restore reconstructs user/tool timestamps; the actual exchange must match.
  const withoutTimestamps = (messages: Message[]) => messages.map(({ timestamp, ...message }) => message);
  assert.deepEqual(withoutTimestamps(restored.sessions.get(result.taskId)!.messages), withoutTimestamps(f.history()));
  const followUp = await runExpertTurn(restored, f.sourceCtx, "Expert instructions.", f.extCtx, { taskId: result.taskId, prompt: "Continue." });
  assert.ok(followUp.ok);
  assert.equal(followUp.text, "RESTORED");
  assert.equal(f.requests.at(-1)!.options?.toolChoice, undefined);
});

test("repeated post-budget calls fail without changing prior history or persistence", async (t) => {
  const f = fixture(t, [answer("FIRST"), calls(8), calls(1, 8), calls(1, 9), calls(1, 10)]);
  await f.run();
  const history = structuredClone(f.history());
  const stored = f.stored();
  const result = await f.run({ taskId: "task-1" });
  assert.equal(result.ok, false);
  if (!result.ok) assert.match(result.error, /budget/i);
  assert.equal(f.requests.length, 5);
  assert.deepEqual(f.history(), history);
  assert.deepEqual(f.stored(), stored);
  for (const request of f.requests) assertPaired(request.context.messages);
});

test("other APIs retain tools and recover without OpenAI-specific options", async (t) => {
  const f = fixture(t, [calls(8), calls(1, 8), answer()], "chronos-test");
  const result = await f.run();
  assert.ok(result.ok);
  assert.equal(result.text, "DONE");
  assert.ok(f.requests.every((r) => r.context.tools!.length > 0 && r.options?.toolChoice === undefined));
});

test("provider errors do not commit a partial turn", async (t) => {
  const f = fixture(t, [calls(1), { ...reply([], "error"), errorMessage: "provider unavailable" }]);
  const result = await f.run();
  assert.equal(result.ok, false);
  if (!result.ok) assert.match(result.error, /provider unavailable/);
  assert.deepEqual(f.history(), []);
  assert.deepEqual(f.stored(), []);
});

test("cancellation before or during completion does not commit a turn", async (t) => {
  const controller = new AbortController();
  const f = fixture(t, [() => { controller.abort(); return answer("PARTIAL"); }]);
  const result = await f.run({ signal: controller.signal });
  assert.equal(result.ok, false);
  assert.deepEqual(f.history(), []);
  assert.deepEqual(f.stored(), []);
  const alreadyAborted = await f.run({ taskId: "task-1", signal: controller.signal });
  assert.equal(alreadyAborted.ok, false);
  assert.equal(f.requests.length, 1);
});

test("provider-reported abortion does not commit a turn", async (t) => {
  const f = fixture(t, [reply([], "aborted")]);
  assert.equal((await f.run()).ok, false);
  assert.deepEqual(f.history(), []);
  assert.deepEqual(f.stored(), []);
});
