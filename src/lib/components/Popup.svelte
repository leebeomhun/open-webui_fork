<script lang="ts">
	import { createEventDispatcher, getContext, onMount } from 'svelte';
	import Modal from './common/Modal.svelte';
	import XMark from '$lib/components/icons/XMark.svelte';
	import popupContent from '$lib/config/popup-content.md?raw';
	import { marked } from 'marked';

	const dispatch = createEventDispatcher();
	const i18n = getContext('i18n');

	export let show = false;

	let config = {
		title: '공지사항',
		content: '',
		buttons: {
			close: '닫기',
			closeForDay: '하루 동안 보지 않기',
			dismissForever: '다시 보지 않기'
		}
	};

	onMount(() => {
		parseMarkdownContent();
	});

	function parseMarkdownContent() {
		try {
			// YAML 설정 부분을 찾기 위해 주석을 기준으로 분리
			const yamlCommentIndex = popupContent.indexOf('<!-- 팝업 설정 (YAML 형태) -->');
			
			if (yamlCommentIndex !== -1) {
				// 주석 이전까지가 마크다운 콘텐츠
				const markdownContent = popupContent.substring(0, yamlCommentIndex).trim();
				
				// 주석 이후 부분에서 YAML 찾기
				const yamlSection = popupContent.substring(yamlCommentIndex);
				const yamlParts = yamlSection.split('---');
				
				if (yamlParts.length >= 2) {
					const yamlContent = yamlParts[1];

					const titleMatch = yamlContent.match(/title:\s*"([^"]+)"/);
					const closeMatch = yamlContent.match(/close:\s*"([^"]+)"/);
					const closeDayMatch = yamlContent.match(/closeForDay:\s*"([^"]+)"/);
					const dismissForeverMatch = yamlContent.match(/dismissForever:\s*"([^"]+)"/);

					if (titleMatch) config.title = titleMatch[1];
					if (closeMatch) config.buttons.close = closeMatch[1];
					if (closeDayMatch) config.buttons.closeForDay = closeDayMatch[1];
					if (dismissForeverMatch) config.buttons.dismissForever = dismissForeverMatch[1];
				}

				config.content = marked(markdownContent) as string;
			} else {
				config.content = marked(popupContent) as string;
			}
		} catch (error) {
			console.error('마크다운 파싱 오류:', error);
			config.content = '내용을 로드할 수 없습니다.';
		}
	}

	function closeModal() {
		show = false;
		dispatch('close');
	}

	function closeForDay() {
		show = false;
		dispatch('close-for-day');
	}

	function dismissPermanently() {
		show = false;
		dispatch('dismiss-permanently');
	}
</script>

<Modal bind:show size="lg">
	<div class="px-5 pt-4 dark:text-gray-300 text-gray-700">
		<div class="flex justify-between items-start">
			<div class="text-xl font-semibold">
				{config.title}
			</div>
			<button class="self-center" on:click={closeModal} aria-label={config.buttons.close}>
				<XMark className={'size-5'}>
					<p class="sr-only">{config.buttons.close}</p>
				</XMark>
			</button>
		</div>
	</div>

	<div class="w-full p-4 px-5 text-gray-700 dark:text-gray-100">
		<div class="overflow-y-scroll max-h-96 scrollbar-hidden">
			<div class="prose dark:prose-invert max-w-none">
				{@html config.content}
			</div>
		</div>
		<div class="flex justify-end pt-3 text-sm font-medium space-x-2">
			<button
				on:click={closeModal}
				class="px-3.5 py-1.5 text-sm font-medium border border-gray-300 bg-white hover:bg-gray-50 text-gray-700 dark:border-gray-600 dark:bg-gray-700 dark:text-gray-200 dark:hover:bg-gray-600 transition rounded-full"
			>
				<span class="relative">{config.buttons.close}</span>
			</button>
			<button
				on:click={closeForDay}
				class="px-3.5 py-1.5 text-sm font-medium bg-sky-600 hover:bg-sky-700 text-white transition rounded-full"
			>
				<span class="relative">{config.buttons.closeForDay}</span>
			</button>
			<button
				on:click={dismissPermanently}
				class="px-3.5 py-1.5 text-sm font-medium bg-black hover:bg-gray-900 text-white dark:bg-white dark:text-black dark:hover:bg-gray-100 transition rounded-full"
			>
				<span class="relative">{config.buttons.dismissForever}</span>
			</button>
		</div>
	</div>
</Modal>

<style>
	:global(.prose h1) {
		font-size: 1.25rem; /* text-xl */
		margin-bottom: 1rem;
	}
	:global(.prose strong) {
		font-size: 1.125rem; /* text-lg */
		font-weight: 600;
	}
	:global(.prose p) {
		margin-top: 0.5rem;
		margin-bottom: 0.5rem;
	}
	:global(.prose ul) {
		margin-top: 0.5rem;
		margin-bottom: 0.5rem;
	}
	:global(.prose hr) {
		border-color: #f3f4f6; /* gray-100 */
		margin-top: 0.5rem;
		margin-bottom: 0.5rem;
	}
	:global(.dark .prose hr) {
		border-color: #2d2b35; /* gray-800 */
	}
	:global(.prose li) {
		font-size: 0.95rem; /* text-sm */
	}
</style>
