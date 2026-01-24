import { defineConfig } from 'vitepress'
import { withMermaid } from 'vitepress-plugin-mermaid'
import dewGrammar from '../../editors/textmate/dew.tmLanguage.json'

export default withMermaid(
  defineConfig({
    vite: {
      optimizeDeps: {
        include: ['mermaid'],
      },
    },
    markdown: {
      languages: [dewGrammar as any],
    },
    title: 'Dew',
    description: 'Minimal expression language with multiple backends',

    base: '/dew/',

    themeConfig: {
      nav: [
        { text: 'Guide', link: '/introduction' },
        { text: 'Backends', link: '/backends/wgsl' },
        { text: 'Playground', link: '/playground/', target: '_self' },
        { text: 'rhi', link: 'https://docs.rhi.zone/' },
      ],

      sidebar: {
        '/': [
          {
            text: 'Guide',
            items: [
              { text: 'Introduction', link: '/introduction' },
              { text: 'Use Cases', link: '/use-cases' },
              { text: 'Integration', link: '/integration' },
              { text: 'Optimization', link: '/optimization' },
            ]
          },
          {
            text: 'Crates',
            items: [
              { text: 'dew-core', link: '/core' },
              { text: 'dew-scalar', link: '/scalar' },
              { text: 'dew-linalg', link: '/linalg' },
              { text: 'dew-complex', link: '/complex' },
              { text: 'dew-quaternion', link: '/quaternion' },
            ]
          },
          {
            text: 'Backends',
            items: [
              { text: 'WGSL', link: '/backends/wgsl' },
              { text: 'GLSL', link: '/backends/glsl' },
              { text: 'OpenCL', link: '/backends/opencl' },
              { text: 'CUDA', link: '/backends/cuda' },
              { text: 'HIP', link: '/backends/hip' },
              { text: 'Rust', link: '/backends/rust' },
              { text: 'C', link: '/backends/c' },
              { text: 'TokenStream', link: '/backends/tokenstream' },
              { text: 'Lua', link: '/backends/lua' },
              { text: 'Cranelift', link: '/backends/cranelift' },
            ]
          },
          {
            text: 'Reference',
            items: [
              { text: 'API Reference', link: '/api-reference' },
            ]
          },
        ]
      },

      socialLinks: [
        { icon: 'github', link: 'https://github.com/rhi-zone/dew' }
      ],

      search: {
        provider: 'local'
      },

      editLink: {
        pattern: 'https://github.com/rhi-zone/dew/edit/master/docs/:path',
        text: 'Edit this page on GitHub'
      },
    },
  }),
)
