<script lang="ts">
	import { getContext, onMount, tick } from 'svelte';
	import Modal from '$lib/components/common/Modal.svelte';
	import Tooltip from '$lib/components/common/Tooltip.svelte';
	import Markdown from '$lib/components/chat/Messages/Markdown.svelte';
	import { WEBUI_API_BASE_URL } from '$lib/constants';
	import { settings } from '$lib/stores';

	import XMark from '$lib/components/icons/XMark.svelte';
	import Textarea from '$lib/components/common/Textarea.svelte';

	const i18n = getContext('i18n');

	const CONTENT_PREVIEW_LIMIT = 10000;
	let expandedDocs: Set<number> = new Set();

	export let show = false;
	export let citation;
	export let showPercentage = false;
	export let showRelevance = true;

	let mergedDocuments = [];

	function calculatePercentage(distance: number) {
		if (typeof distance !== 'number') return null;
		if (distance < 0) return 0;
		if (distance > 1) return 100;
		return Math.round(distance * 10000) / 100;
	}

	function getRelevanceColor(percentage: number) {
		if (percentage >= 80)
			return 'bg-green-200 dark:bg-green-800 text-green-800 dark:text-green-200';
		if (percentage >= 60)
			return 'bg-yellow-200 dark:bg-yellow-800 text-yellow-800 dark:text-yellow-200';
		if (percentage >= 40)
			return 'bg-orange-200 dark:bg-orange-800 text-orange-800 dark:text-orange-200';
		return 'bg-red-200 dark:bg-red-800 text-red-800 dark:text-red-200';
	}

	$: if (citation) {
		expandedDocs = new Set();
		mergedDocuments = citation.document?.map((c, i) => {
			return {
				source: citation.source,
				document: c,
				metadata: citation.metadata?.[i],
				distance: citation.distances?.[i]
			};
		});
		if (mergedDocuments.every((doc) => doc.distance !== undefined)) {
			mergedDocuments = mergedDocuments.sort(
				(a, b) => (b.distance ?? Infinity) - (a.distance ?? Infinity)
			);
		}
	}

	const decodeString = (str: string) => {
		try {
			return decodeURIComponent(str);
		} catch {
			return str;
		}
	};

	const getTextFragmentUrl = (doc: any): string | null => {
		const { metadata, source, document: content } = doc ?? {};
		const { file_id, page } = metadata ?? {};
		const sourceUrl = source?.url;

		const baseUrl = file_id
			? `${WEBUI_API_BASE_URL}/files/${file_id}/content${page !== undefined ? `#page=${page + 1}` : ''}`
			: sourceUrl?.includes('http')
				? sourceUrl
				: null;

		if (!baseUrl || !content) return baseUrl;

		// Extract first and last words for text fragment, filtering out URLs and emojis
		const words = content
			.trim()
			.replace(/\s+/g, ' ')
			.split(' ')
			.filter((w: string) => w.length > 0 && !/https?:\/\/|[\u{1F300}-\u{1F9FF}]/u.test(w));

		if (words.length === 0) return baseUrl;

		const clean = (w: string) => w.replace(/[^\w]/g, '');
		const first = clean(words[0]);
		const last = clean(words.at(-1));
		const fragment = words.length === 1 ? first : `${first},${last}`;

		return fragment ? `${baseUrl}#:~:text=${fragment}` : baseUrl;
	};

	// 표시용 콘텐츠에서 리터럴 "\n"을 실제 줄바꿈으로 변환
	const formatContent = (value: unknown): string => {
		if (value === null || value === undefined) return '';
		const str = String(value);
		return str
			.replace(/\\r\\n/g, '\n') // 리터럴 "\r\n" -> 실제 개행
			.replace(/\r\n/g, '\n') // 실제 CRLF -> LF 통일
			.replace(/\\n/g, '\n'); // 리터럴 "\n" -> 실제 개행
	};

	// JSON 형태 콘텐츠 파싱 시도 (문자열/객체 모두 처리)
	const parseContent = (value: unknown): any | null => {
		try {
			if (!value) return null;
			if (typeof value === 'object') return value as any;
			if (typeof value === 'string') {
				const trimmed = value.trim();
				if ((trimmed.startsWith('{') && trimmed.endsWith('}')) || (trimmed.startsWith('[') && trimmed.endsWith(']'))) {
					return JSON.parse(trimmed);
				}
			}
		} catch (e) {
			return null;
		}
		return null;
	};

	// 구조화된 결과 여부 판단
	const isStructuredContent = (obj: any): boolean => {
		if (!obj || typeof obj !== 'object') return false;
		return (
			Array.isArray(obj.expanded_queries) ||
			(obj.kcd_by_query && typeof obj.kcd_by_query === 'object') ||
			(obj.results && typeof obj.results === 'object')
		);
	};

	// kcd_by_query 객체를 단일 배열로 평탄화
	const flattenKcd = (kcd: any): any[] => {
		if (!kcd || typeof kcd !== 'object') return [];
		const list: any[] = [];
		for (const key of Object.keys(kcd)) {
			const items = kcd[key];
			if (Array.isArray(items)) {
				for (const it of items) list.push(it);
			}
		}
		return list;
	};

	// 점수(0~1)를 퍼센트 문자열로 변환 (소수점 둘째자리)
	const toPercentStr = (score: unknown): string => {
		const n = typeof score === 'number' ? score : Number(score);
		if (Number.isFinite(n)) return `${(n * 100).toFixed(2)}%`;
		return '';
	};
</script>

<Modal size="lg" bind:show>
	<div>
		<div class=" flex justify-between dark:text-gray-300 px-4.5 pt-3 pb-2">
			<div class=" text-lg font-medium self-center flex items-center">
				{#if citation?.source?.name}
					{@const document = mergedDocuments?.[0]}
					{#if document?.metadata?.file_id || document.source?.url?.includes('http')}
						<Tooltip
							className="w-fit"
							content={document.source?.url?.includes('http')
								? $i18n.t('Open link')
								: $i18n.t('Open file')}
							placement="top-start"
							tippyOptions={{ duration: [500, 0] }}
						>
							<a
								class="hover:text-gray-500 dark:hover:text-gray-100 underline grow line-clamp-1"
								href={document?.metadata?.file_id
									? `${WEBUI_API_BASE_URL}/files/${document?.metadata?.file_id}/content${document?.metadata?.page !== undefined ? `#page=${document.metadata.page + 1}` : ''}`
									: document.source?.url?.includes('http')
										? document.source.url
										: `#`}
								target="_blank"
							>
								{decodeString(citation?.source?.name)}
							</a>
						</Tooltip>
					{:else}
						{decodeString(citation?.source?.name)}
					{/if}
				{:else}
					{$i18n.t('Citation')}
				{/if}
			</div>
			<button
				class="self-center"
				aria-label={$i18n.t('Close citation modal')}
				on:click={() => {
					show = false;
				}}
			>
				<XMark className={'size-5'} />
			</button>
		</div>

		<div class="flex flex-col md:flex-row w-full px-5 pb-5 md:space-x-4">
			<div
				class="flex flex-col w-full dark:text-gray-200 overflow-y-scroll max-h-[22rem] scrollbar-thin gap-1"
			>
				{#each mergedDocuments as document, documentIdx}
					<div class="flex flex-col w-full gap-2">
						{#if document.metadata?.parameters}
							<div>
								<div class="text-sm font-medium dark:text-gray-300 mb-1">
									{$i18n.t('Parameters')}
								</div>

								<Textarea readonly value={JSON.stringify(document.metadata.parameters?.query, null, 2)}
								></Textarea>
							</div>
						{/if}

						<div>
							<div
								class=" text-sm font-medium dark:text-gray-300 flex items-center gap-2 w-fit mb-1"
							>
								{#if document.source?.url?.includes('http')}
									{@const snippetUrl = getTextFragmentUrl(document)}
									{#if snippetUrl}
										<a
											href={snippetUrl}
											target="_blank"
											class="underline hover:text-gray-500 dark:hover:text-gray-100"
											>{$i18n.t('Content')}</a
										>
									{:else}
										{$i18n.t('Content')}
									{/if}
								{:else}
									{$i18n.t('Content')}
								{/if}

								{#if showRelevance && document.distance !== undefined}
									<Tooltip
										className="w-fit"
										content={$i18n.t('Relevance')}
										placement="top-start"
										tippyOptions={{ duration: [500, 0] }}
									>
										<div class="text-sm my-1 dark:text-gray-400 flex items-center gap-2 w-fit">
											{#if showPercentage}
												{@const percentage = calculatePercentage(document.distance)}

												{#if typeof percentage === 'number'}
													<span
														class={`px-1 rounded-sm font-medium ${getRelevanceColor(percentage)}`}
													>
														{percentage.toFixed(2)}%
													</span>
												{/if}
											{:else if typeof document?.distance === 'number'}
												<span class="text-gray-500 dark:text-gray-500">
													({(document?.distance ?? 0).toFixed(4)})
												</span>
											{/if}
										</div>
									</Tooltip>
								{/if}

								{#if Number.isInteger(document?.metadata?.page)}
									<span class="text-sm text-gray-500 dark:text-gray-400">
										({$i18n.t('page')}
										{document.metadata.page + 1})
									</span>
								{/if}
							</div>

							{#if document.metadata?.html}
								<iframe
									class="w-full border-0 h-auto rounded-none"
									sandbox="allow-scripts allow-forms{($settings?.iframeSandboxAllowSameOrigin ??
									false)
										? ' allow-same-origin'
										: ''}"
									srcdoc={document.document}
									title={$i18n.t('Content')}
								></iframe>
							{:else}
								{@const parsed = parseContent(document.document)}
								{#if isStructuredContent(parsed)}
									{#if Array.isArray(parsed?.expanded_queries) && parsed.expanded_queries.length}
										<div class="text-sm font-medium dark:text-gray-300 mt-2">확장된 쿼리</div>
										<ul class="list-disc pl-5 dark:text-gray-400">
											{#each parsed.expanded_queries as q}
												<li>{q}</li>
											{/each}
										</ul>
									{/if}

									{#if parsed?.kcd_by_query}
										{@const kcdList = flattenKcd(parsed.kcd_by_query)}
										{#if kcdList.length}
											<div class="text-sm font-medium dark:text-gray-300 mt-3">검색 결과</div>
											<div class="space-y-3">
												{#each kcdList as item}
													<div class="rounded-md border border-gray-200 dark:border-gray-800 p-3">
														<div class="flex flex-wrap gap-x-4 gap-y-1 text-sm dark:text-gray-400">
															<div><span class="font-medium">출처:</span> {item?.file_name ?? '-'}</div>
															{#if item?.score !== undefined}
																<div><span class="font-medium">관련도:</span> {toPercentStr(item.score)}</div>
															{/if}
														</div>
														{#if item?.text}
															<div class="mt-2 text-sm dark:text-gray-300">
																<div class="font-medium mb-1">내용</div>
																<pre class="whitespace-pre-line text-[13px] leading-5 dark:text-gray-400">{formatContent(item.text)}</pre>
															</div>
														{/if}
													</div>
												{/each}
											</div>
										{/if}
									{/if}

									{#if parsed?.results}
										<div class="text-sm font-medium dark:text-gray-300 mt-3">추가 정보</div>
										<div class="space-y-2 text-sm dark:text-gray-400">
											{#if Array.isArray(parsed.results?.pathogen) && parsed.results.pathogen.length}
												<div>
													<div class="font-medium dark:text-gray-300">원인균 추가정보</div>
													<ul class="list-disc pl-5">
														{#each parsed.results.pathogen as p}
															<li class="whitespace-pre-line text-[12px] leading-5 dark:text-gray-400">{formatContent(typeof p === 'string' ? p : p?.text ?? JSON.stringify(p))}</li>
													{/each}
												</ul>
											</div>
										{/if}
										{#if Array.isArray(parsed.results?.resistance) && parsed.results.resistance.length}
											<div>
												<div class="font-medium dark:text-gray-300">항생제내성 추가정보</div>
												<ul class="list-disc pl-5">
													{#each parsed.results.resistance as r}
														<li class="whitespace-pre-line text-[12px] leading-5 dark:text-gray-400">{formatContent(typeof r === 'string' ? r : r?.text ?? JSON.stringify(r))}</li>
												{/each}
												</ul>
											</div>
										{/if}
										{#if Array.isArray(parsed.results?.external) && parsed.results.external.length}
											<div>
												<div class="font-medium dark:text-gray-300">외인 추가정보</div>
												<ul class="list-disc pl-5">
													{#each parsed.results.external as e}
														<li class="whitespace-pre-line text-[12px] leading-5 dark:text-gray-400">{formatContent(typeof e === 'string' ? e : e?.text ?? JSON.stringify(e))}</li>
												{/each}
												</ul>
												{#if parsed?.external_note}
													<div class="mt-1 text-xs dark:text-gray-500">외인 3,4단위 적용정보:</div>
													<pre class="whitespace-pre-line mt-1 text-xs dark:text-gray-500">{formatContent(parsed.external_note)}</pre>
												{/if}
											</div>
										{/if}
										</div>
									{/if}
							{:else}
								{@const rawContent = document.document.trim().replace(/\n\n+/g, '\n\n')}
								{@const isTruncated =
									($settings?.renderMarkdownInPreviews ?? true) &&
									rawContent.length > CONTENT_PREVIEW_LIMIT &&
									!expandedDocs.has(documentIdx)}
								{#if $settings?.renderMarkdownInPreviews ?? true}
									<div class="text-sm prose dark:prose-invert max-w-full">
										<Markdown
											content={isTruncated
												? rawContent.slice(0, CONTENT_PREVIEW_LIMIT)
												: rawContent}
											id="citation-{documentIdx}"
										/>
									</div>
									{#if isTruncated}
										<button
											class="mt-1 text-xs text-gray-500 hover:text-gray-700 dark:hover:text-gray-300 transition"
											on:click={() => {
												expandedDocs.add(documentIdx);
												expandedDocs = expandedDocs;
										}}
										>
											{$i18n.t('Show all ({{COUNT}} characters)', {
												COUNT: rawContent.length.toLocaleString()
											})}
										</button>
									{/if}
								{:else}
									<pre class="text-sm dark:text-gray-400 whitespace-pre-line">{rawContent}</pre>
								{/if}
							{/if}
						{/if}
						</div>
					</div>
					{/each}
				</div>
			</div>
		</div>
	</Modal>
