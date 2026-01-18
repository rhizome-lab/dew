import { createSignal, createMemo, createResource, Show, For } from 'solid-js';
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

type CodeResult =
  | { ok: true; code: string }
  | { ok: false; error: string };

type Profile = 'scalar' | 'linalg' | 'complex' | 'quaternion';

type VarTypes = Record<string, string>;

// Profile metadata
const PROFILES: { id: Profile; label: string; description: string; exampleTypes: string }[] = [
  { id: 'scalar', label: 'Scalar', description: 'Basic math', exampleTypes: '' },
  { id: 'linalg', label: 'Linalg', description: 'Vectors & matrices', exampleTypes: '{"v": "vec3", "m": "mat4"}' },
  { id: 'complex', label: 'Complex', description: 'Complex numbers', exampleTypes: '{"z": "complex", "w": "complex"}' },
  { id: 'quaternion', label: 'Quaternion', description: 'Rotations', exampleTypes: '{"q": "quat", "v": "vec3"}' },
];

// WASM module interface (full profile with all exports)
interface DewWasm {
  parse: (input: string) => ParseResult;
  // Scalar
  emit_wgsl: (input: string) => CodeResult;
  emit_glsl: (input: string) => CodeResult;
  emit_lua: (input: string) => CodeResult;
  // Linalg
  emit_wgsl_linalg?: (input: string, varTypes: VarTypes) => CodeResult;
  emit_glsl_linalg?: (input: string, varTypes: VarTypes) => CodeResult;
  emit_lua_linalg?: (input: string, varTypes: VarTypes) => CodeResult;
  // Complex
  emit_wgsl_complex?: (input: string, varTypes: VarTypes) => CodeResult;
  emit_glsl_complex?: (input: string, varTypes: VarTypes) => CodeResult;
  emit_lua_complex?: (input: string, varTypes: VarTypes) => CodeResult;
  // Quaternion
  emit_wgsl_quaternion?: (input: string, varTypes: VarTypes) => CodeResult;
  emit_glsl_quaternion?: (input: string, varTypes: VarTypes) => CodeResult;
  emit_lua_quaternion?: (input: string, varTypes: VarTypes) => CodeResult;
}

// Load WASM module
async function loadWasm(): Promise<DewWasm | null> {
  try {
    const wasm = await import('./wasm/rhizome_dew_wasm.js');
    await wasm.default();
    return {
      parse: wasm.parse,
      // Scalar
      emit_wgsl: wasm.emit_wgsl,
      emit_glsl: wasm.emit_glsl,
      emit_lua: wasm.emit_lua,
      // Linalg (optional)
      emit_wgsl_linalg: wasm.emit_wgsl_linalg,
      emit_glsl_linalg: wasm.emit_glsl_linalg,
      emit_lua_linalg: wasm.emit_lua_linalg,
      // Complex (optional)
      emit_wgsl_complex: wasm.emit_wgsl_complex,
      emit_glsl_complex: wasm.emit_glsl_complex,
      emit_lua_complex: wasm.emit_lua_complex,
      // Quaternion (optional)
      emit_wgsl_quaternion: wasm.emit_wgsl_quaternion,
      emit_glsl_quaternion: wasm.emit_glsl_quaternion,
      emit_lua_quaternion: wasm.emit_lua_quaternion,
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
  return {
    ok: true,
    ast: { type: 'Mock', value: input }
  };
}

function parseVarTypes(input: string): VarTypes | null {
  if (!input.trim()) return {};
  try {
    const parsed = JSON.parse(input);
    if (typeof parsed !== 'object' || parsed === null || Array.isArray(parsed)) {
      return null;
    }
    return parsed as VarTypes;
  } catch {
    return null;
  }
}

export function App() {
  const [expression, setExpression] = createSignal('sin(x) + cos(y) * 2');
  const [profile, setProfile] = createSignal<Profile>('scalar');
  const [varTypesInput, setVarTypesInput] = createSignal('');
  const [activeTab, setActiveTab] = createSignal<'ast' | 'wgsl' | 'glsl' | 'lua'>('ast');
  const [wasm] = createResource(loadWasm);

  const varTypes = createMemo(() => parseVarTypes(varTypesInput()));
  const varTypesError = createMemo(() => {
    if (!varTypesInput().trim()) return null;
    return varTypes() === null ? 'Invalid JSON' : null;
  });

  const needsVarTypes = createMemo(() => profile() !== 'scalar');

  const parseResult = createMemo((): ParseResult => {
    const w = wasm();
    if (w) {
      return w.parse(expression());
    }
    return mockParse(expression());
  });

  // Helper to emit code based on profile
  const emitCode = (
    backend: 'wgsl' | 'glsl' | 'lua',
    w: DewWasm,
    expr: string,
    prof: Profile,
    types: VarTypes | null
  ): CodeResult => {
    if (prof === 'scalar') {
      const fn = backend === 'wgsl' ? w.emit_wgsl : backend === 'glsl' ? w.emit_glsl : w.emit_lua;
      return fn(expr);
    }

    if (types === null) {
      return { ok: false, error: 'Invalid variable types JSON' };
    }

    const fnMap = {
      linalg: { wgsl: w.emit_wgsl_linalg, glsl: w.emit_glsl_linalg, lua: w.emit_lua_linalg },
      complex: { wgsl: w.emit_wgsl_complex, glsl: w.emit_glsl_complex, lua: w.emit_lua_complex },
      quaternion: { wgsl: w.emit_wgsl_quaternion, glsl: w.emit_glsl_quaternion, lua: w.emit_lua_quaternion },
    };

    const fn = fnMap[prof]?.[backend];
    if (!fn) {
      return { ok: false, error: `${prof} backend not available in this WASM build` };
    }

    return fn(expr, types);
  };

  const wgslResult = createMemo((): CodeResult => {
    const w = wasm();
    if (!w) return { ok: false, error: 'WASM not loaded' };
    return emitCode('wgsl', w, expression(), profile(), varTypes());
  });

  const glslResult = createMemo((): CodeResult => {
    const w = wasm();
    if (!w) return { ok: false, error: 'WASM not loaded' };
    return emitCode('glsl', w, expression(), profile(), varTypes());
  });

  const luaResult = createMemo((): CodeResult => {
    const w = wasm();
    if (!w) return { ok: false, error: 'WASM not loaded' };
    return emitCode('lua', w, expression(), profile(), varTypes());
  });

  const setProfileWithExample = (p: Profile) => {
    setProfile(p);
    const meta = PROFILES.find(pr => pr.id === p);
    if (meta?.exampleTypes && !varTypesInput()) {
      setVarTypesInput(meta.exampleTypes);
    }
  };

  return (
    <div class="playground">
      <header class="playground__header">
        <a href="/dew/" class="playground__back">&larr; Docs</a>
        <h1 class="playground__title">Dew Playground</h1>
        <div class="header__controls">
          <select
            class="profile-select"
            value={profile()}
            onChange={(e) => setProfileWithExample(e.currentTarget.value as Profile)}
          >
            <For each={PROFILES}>
              {(p) => (
                <option value={p.id}>{p.label}</option>
              )}
            </For>
          </select>
          <Show when={wasm.loading}>
            <span class="loading-indicator">Loading WASM...</span>
          </Show>
          <Show when={!wasm.loading && !wasm()}>
            <span class="loading-indicator loading-indicator--warn">WASM unavailable</span>
          </Show>
        </div>
      </header>

      <main class="playground__main">
        <div class="panel">
          <div class="panel__header">
            <span>Expression</span>
            <span class="profile-badge">{PROFILES.find(p => p.id === profile())?.description}</span>
          </div>
          <div class="panel__content panel__content--no-padding">
            <div class="editor-container">
              <Editor
                value={expression()}
                onChange={setExpression}
                placeholder="Enter a dew expression..."
              />
              <Show when={needsVarTypes()}>
                <div class="var-types-section">
                  <label class="var-types-label">
                    Variable Types
                    <Show when={varTypesError()}>
                      <span class="var-types-error">{varTypesError()}</span>
                    </Show>
                  </label>
                  <input
                    type="text"
                    class="var-types-input"
                    classList={{ 'var-types-input--error': !!varTypesError() }}
                    value={varTypesInput()}
                    onInput={(e) => setVarTypesInput(e.currentTarget.value)}
                    placeholder='{"x": "vec3", "y": "vec3"}'
                  />
                </div>
              </Show>
            </div>
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
              <button
                class={`tabs__tab ${activeTab() === 'glsl' ? 'tabs__tab--active' : ''}`}
                onClick={() => setActiveTab('glsl')}
              >
                GLSL
              </button>
              <button
                class={`tabs__tab ${activeTab() === 'lua' ? 'tabs__tab--active' : ''}`}
                onClick={() => setActiveTab('lua')}
              >
                Lua
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
              <Show when={wgslResult().ok} fallback={
                <div class="output__value output__value--error">
                  {(wgslResult() as { ok: false; error: string }).error}
                </div>
              }>
                <pre class="code-block">{(wgslResult() as { ok: true; code: string }).code}</pre>
              </Show>
            </Show>
            <Show when={activeTab() === 'glsl'}>
              <Show when={glslResult().ok} fallback={
                <div class="output__value output__value--error">
                  {(glslResult() as { ok: false; error: string }).error}
                </div>
              }>
                <pre class="code-block">{(glslResult() as { ok: true; code: string }).code}</pre>
              </Show>
            </Show>
            <Show when={activeTab() === 'lua'}>
              <Show when={luaResult().ok} fallback={
                <div class="output__value output__value--error">
                  {(luaResult() as { ok: false; error: string }).error}
                </div>
              }>
                <pre class="code-block">{(luaResult() as { ok: true; code: string }).code}</pre>
              </Show>
            </Show>
          </div>
        </div>
      </main>
    </div>
  );
}
