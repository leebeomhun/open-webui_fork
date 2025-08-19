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

	// ChangelogModal.svelte와 동일한 렌더 구조를 사용하기 위한 파싱 결과
	let parsedChangelog: Record<string, any> | null = null;
	let hasStructuredContent = false;

	onMount(() => {
		parseMarkdownContent();
	});

	function parseMarkdownContent() {
		try {
			// YAML 설정 부분을 찾기 위해 주석을 기준으로 분리
			const yamlCommentIndex = popupContent.indexOf('<!-- 팝업 설정 (YAML 형태) -->');
			let markdownContent = popupContent;
			if (yamlCommentIndex !== -1) {
				markdownContent = popupContent.substring(0, yamlCommentIndex).trim();
				// 주석 이후 부분에서 YAML 찾기 및 버튼/타이틀만 반영
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
			}

			// 마크다운을 변경 로그 구조로 파싱 시도
			parsedChangelog = parseMarkdownToChangelog(markdownContent);
			hasStructuredContent = !!parsedChangelog && Object.keys(parsedChangelog).length > 0;

			// 파싱 실패 시 기존 마크다운 렌더링으로 폴백
			if (!hasStructuredContent) {
				config.content = marked(markdownContent) as string;
			}
		} catch (error) {
			console.error('마크다운 파싱 오류:', error);
			config.content = '내용을 로드할 수 없습니다.';
		}
	}

	function parseMarkdownToChangelog(md: string): Record<string, any> | null {
		const lines = md.split(/\r?\n/);
		const result: Record<string, any> = {};
		let currentVersion: string | null = null;
		let currentDate: string | null = null;
		let currentSection: string | null = null;
		let sectionIndex = 0;

		const versionRe = /^##\s*\[(.*?)\]\s*-\s*(.*)$/i; // ## [0.6.22] - 2025-08-11
		const sectionRe = /^###\s*(.+)$/i; // ### Added
		const bulletRe = /^-\s+(.*)$/; // - ...

		for (let i = 0; i < lines.length; i++) {
			const raw = lines[i];
			const line = raw.trim();
			if (!line) continue;

			const vMatch = line.match(versionRe);
			if (vMatch) {
				currentVersion = vMatch[1].trim();
				currentDate = vMatch[2].trim();
				result[currentVersion] = { date: currentDate };
				currentSection = null;
				sectionIndex = 0;
				continue;
			}

			const sMatch = line.match(sectionRe);
			if (sMatch && currentVersion) {
				currentSection = normalizeSection(sMatch[1]);
				if (!result[currentVersion][currentSection]) result[currentVersion][currentSection] = {};
				sectionIndex = 0;
				continue;
			}

			const bMatch = line.match(bulletRe);
			if (bMatch && currentVersion && currentSection) {
				const text = bMatch[1].trim();
				let title = '';
				let content = '';

				// 굵은 제목 추출: **title**: content, 이모지 등 접두 포함
				const strongMatch = text.match(/\*\*(.+?)\*\*/);
				if (strongMatch) {
					const boldToken = strongMatch[0];
					const boldIndex = text.indexOf(boldToken);
					const prefix = text.slice(0, boldIndex).trim();
					title = `${prefix ? prefix + ' ' : ''}${strongMatch[1].trim()}`.trim();
					const after = text.slice(boldIndex + boldToken.length).trim();
					content = after.replace(/^:\s*/, '').trim();
				} else {
					// 첫 콜론 기준 분리
					const idx = text.indexOf(':');
					if (idx !== -1) {
						title = text.slice(0, idx).trim();
						content = text.slice(idx + 1).trim();
					} else {
						title = text;
						content = '';
					}
				}

				// 이어지는 들여쓰기/빈 줄을 본문에 포함 (마크다운 스타일 여러 줄)
				while (i + 1 < lines.length) {
					const lookaheadRaw = lines[i + 1];
					const lookaheadTrim = lookaheadRaw.trim();

					// 다음 섹션/버전/새 불릿이면 중단
					if (lookaheadTrim.match(versionRe) || lookaheadTrim.match(sectionRe) || lookaheadTrim.match(bulletRe)) break;

					// 동일 항목 본문 계속
					i++;
					if (!lookaheadTrim) {
						content += (content ? '<br>' : '') + '';
						continue;
					}
					// 앞쪽 들여쓰기 제거 후 추가
					const cont = lookaheadRaw.replace(/^\s{1,}/, '').trim();
					content += (content ? '<br>' : '') + cont;
				}

				// 인덱스 키 사용(ChangelogModal 구조에 맞춤)
				const key = String(sectionIndex++);
				result[currentVersion][currentSection][key] = { title, content };
				continue;
			}
		}

		return Object.keys(result).length ? result : null;
	}

	function normalizeSection(s: string): string {
		const n = s.trim().toLowerCase();
		if (n.includes('add')) return '업데이트';
		if (n.includes('fix')) return 'fixed';
		if (n.includes('change')) return 'changed';
		if (n.includes('remove')) return 'removed';
		return n;
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
			{#if hasStructuredContent && parsedChangelog}
				<div class="mb-3">
					{#each Object.keys(parsedChangelog) as version}
						<div class=" mb-3 pr-2">
							<div class="font-semibold text-xl mb-1 dark:text-white">
								{version} - {parsedChangelog[version].date}
							</div>

							<hr class="border-gray-100 dark:border-gray-850 my-2" />

							{#each Object.keys(parsedChangelog[version]).filter((section) => section !== 'date') as section}
								<div>
									<div
										class="font-semibold uppercase text-xs {section === '업데이트'
											? 'text-white bg-blue-600'
											: section === 'fixed'
												? 'text-white bg-green-600'
												: section === 'changed'
													? 'text-white bg-yellow-600'
													: section === 'removed'
														? 'text-white bg-red-600'
														: ''}  w-fit px-3 rounded-full my-2.5"
									>
										{section}
									</div>

									<div class="my-2.5 px-1.5">
										{#each Object.keys(parsedChangelog[version][section]) as item}
											<div class="text-sm mb-2">
											<div class="font-semibold uppercase text-base">
												{parsedChangelog[version][section][item].title}
											</div>
											<div class="mb-2 mt-1">
												{@html parsedChangelog[version][section][item].content}
											</div>
											</div>
										{/each}
									</div>
								</div>
							{/each}
						</div>
					{/each}
				</div>
			{:else}
				<div class="prose dark:prose-invert max-w-none">
					{@html config.content}
				</div>
			{/if}
		</div>
		<div class="flex justify-end pt-3 text-sm font-medium space-x-1 sm:space-x-2">
			<button
				on:click={closeModal}
				class="px-2 py-1.5 text-xs sm:px-3.5 sm:text-sm font-medium border border-gray-300 bg-white hover:bg-gray-50 text-gray-700 dark:border-gray-600 dark:bg-gray-700 dark:text-gray-200 dark:hover:bg-gray-600 transition rounded-full whitespace-nowrap"
			>
				<span class="relative">{config.buttons.close}</span>
			</button>
			<button
				on:click={closeForDay}
				class="px-2 py-1.5 text-xs sm:px-3.5 sm:text-sm font-medium bg-sky-600 hover:bg-sky-700 text-white transition rounded-full whitespace-nowrap"
			>
				<span class="relative">{config.buttons.closeForDay}</span>
			</button>
			<button
				on:click={dismissPermanently}
				class="px-2 py-1.5 text-xs sm:px-3.5 sm:text-sm font-medium bg-black hover:bg-gray-900 text-white dark:bg-white dark:text-black dark:hover:bg-gray-100 transition rounded-full whitespace-nowrap"
			>
				<span class="relative">{config.buttons.dismissForever}</span>
			</button>
		</div>
	</div>
</Modal>

<!-- <style>
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
		margin-bottom: 0.125rem;
		line-height: 1.5;
	}
</style> -->
