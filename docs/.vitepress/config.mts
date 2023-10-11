import {defineConfig} from 'vitepress'

// https://vitepress.dev/reference/site-config
export default defineConfig({
    title: "ggml",
    description: "Tensor library for machine learning",
    themeConfig: {
        // https://vitepress.dev/reference/default-theme-config
        nav: [
            {text: 'Guide', link: '/guide/get-start'},
            {text: 'Reference', link: '/reference/markdown-examples'}
        ],
        sidebar: {
            '/guide/': [
                {
                text: 'Introduction',
                items: [
                    {text: 'Getting Started', link: '/guide/get-start'},
                ]
            }
            ],
            '/reference/': [{
                text: 'Reference',
                items: [
                    {text: 'Markdown Examples', link: '/markdown-examples'},
                    {text: 'Runtime API Examples', link: '/api-examples'}
                ]
            }]
        },

        socialLinks: [
            {icon: 'github', link: 'https://github.com/ggerganov/ggml'}
        ]
    }
})
