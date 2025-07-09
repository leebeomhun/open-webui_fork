import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';
import { createHash } from 'crypto'; 
import { readFileSync } from 'fs';

import { viteStaticCopy } from 'vite-plugin-static-copy';
const popupContent = readFileSync('src/lib/config/popup-content.md', 'utf-8');
const popupHash = createHash('sha256').update(popupContent).digest('hex');
export default defineConfig({
	plugins: [
		sveltekit(),
		viteStaticCopy({
			targets: [
				{
					src: 'node_modules/onnxruntime-web/dist/*.jsep.*',

					dest: 'wasm'
				}
			]
		})
	],
	define: {
		APP_VERSION: JSON.stringify(process.env.npm_package_version),
		APP_BUILD_HASH: JSON.stringify(process.env.APP_BUILD_HASH || 'dev-build'),
		POPUP_HASH: JSON.stringify(popupHash)
	},
	build: {
		sourcemap: true
	},
	worker: {
		format: 'es'
	},
	esbuild: {
		pure: process.env.ENV === 'dev' ? [] : ['console.log', 'console.debug', 'console.error']
	},
	server: {
		host: true,
		allowedHosts: ['kcdassistant.duckdns.org']
	}
});
