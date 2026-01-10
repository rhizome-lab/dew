import { createSignal, createMemo, createResource, Show, onMount } from 'solid-js';
import { Editor } from './components/Editor';
import { AstViewer } from './components/AstViewer';

// Types matching WASM output
type AstNode = {
  type: string;
  value?: string;
  children?: AstNode[];
};

type ParseResult =
  | { ok: true; ast: AstNode }
  | { ok: false; error: string };

type WgslResult =
  | { ok: true; code: string }
  | { ok: false; error: string };

// WASM module interface
interface DewWasm {
  parse: (input: string) => ParseResult;
  emit_wgsl: (input: string) => WgslResult;
}

// Load WASM module
async function loadWasm(): Promise<DewWasm | null> {
  try {
    const wasm = await import('./wasm/rhizome_dew_wasm.js');
    await wasm.default();
    return {
      parse: wasm.parse,
      emit_wgsl: wasm.emit_wgsl,
    };
  } catch (e) {
    console.warn('WASM not available, using mock parser:', e);
    return null;
  }
}

// Fallback mock parser for local dev without WASM
function mockParse(input: string): ParseResult {
  if (!input.trim()) {
    return { ok: false, error: 'Empty expression' };
  }
  // Very basic mock for development
  return {
    ok: true,
    ast: { type: 'Mock', value: input }
  };
}

export function App() {
  const [expression, setExpression] = createSignal('sin(x) + cos(y) * 2');
  const [activeTab, setActiveTab] = createSignal<'ast' | 'wgsl'>('ast');
  const [wasm] = createResource(loadWasm);

  const parseResult = createMemo((): ParseResult => {
    const w = wasm();
    if (w) {
      return w.parse(expression());
    }
    return mockParse(expression());
  });

  const wgslResult = createMemo((): WgslResult | null => {
    const w = wasm();
    if (!w) return { ok: false, error: 'WASM not loaded' };
    return w.emit_wgsl(expression());
  });

  return (
    <div class="playground">
      <header class="playground__header">
        <h1 class="playground__title">Dew Playground</h1>
        <Show when={wasm.loading}>
          <span class="loading-indicator">Loading WASM...</span>
        </Show>
        <Show when={!wasm.loading && !wasm()}>
          <span class="loading-indicator loading-indicator--warn">WASM unavailable (mock mode)</span>
        </Show>
      </header>

      <main class="playground__main">
        <div class="panel">
          <div class="panel__header">
            Expression
          </div>
          <div class="panel__content panel__content--no-padding">
            <Editor
              value={expression()}
              onChange={setExpression}
              placeholder="Enter a dew expression..."
            />
          </div>
        </div>

        <div class="panel">
          <div class="panel__header">
            <span>Output</span>
            <div class="tabs">
              <button
                class={`tabs__tab ${activeTab() === 'ast' ? 'tabs__tab--active' : ''}`}
                onClick={() => setActiveTab('ast')}
              >
                AST
              </button>
              <button
                class={`tabs__tab ${activeTab() === 'wgsl' ? 'tabs__tab--active' : ''}`}
                onClick={() => setActiveTab('wgsl')}
              >
                WGSL
              </button>
            </div>
          </div>
          <div class="panel__content">
            <Show when={activeTab() === 'ast'}>
              <Show when={parseResult().ok} fallback={
                <div class="output__value output__value--error">
                  {(parseResult() as { ok: false; error: string }).error}
                </div>
              }>
                <AstViewer ast={(parseResult() as { ok: true; ast: AstNode }).ast} />
              </Show>
            </Show>
            <Show when={activeTab() === 'wgsl'}>
              <Show when={wgslResult()?.ok} fallback={
                <div class="output__value output__value--error">
                  {(wgslResult() as { ok: false; error: string })?.error || 'Error generating WGSL'}
                </div>
              }>
                <pre class="code-block">{(wgslResult() as { ok: true; code: string }).code}</pre>
              </Show>
            </Show>
          </div>
        </div>
      </main>
    </div>
  );
}
