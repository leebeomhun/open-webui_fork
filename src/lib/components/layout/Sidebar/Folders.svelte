<script lang="ts">
	import { createEventDispatcher, getContext } from 'svelte';
	const dispatch = createEventDispatcher();
	const i18n = getContext('i18n');

	import RecursiveFolder from './RecursiveFolder.svelte';
	import { chatId, selectedFolder } from '$lib/stores';
	import ChevronDown from '../../icons/ChevronDown.svelte';
	import ChevronUp from '../../icons/ChevronUp.svelte';

	export let folderRegistry = {};

	export let folders = {};
	export let shiftKey = false;

	export let onDelete = (folderId) => {};

	const MAX_VISIBLE_FOLDERS = 5;
	let showAllFolders = false;

	let folderList = [];
	// Get the list of folders that have no parent, sorted by last_chat_updated_at (most recent first)
	$: folderList = Object.keys(folders)
		.filter((key) => folders[key].parent_id === null)
		.sort((a, b) => {
			// Sort by last_chat_updated_at in descending order (most recent chat first)
			// If no chats in folder, fall back to folder's updated_at
			const aLastChat = folders[a].last_chat_updated_at || folders[a].updated_at || 0;
			const bLastChat = folders[b].last_chat_updated_at || folders[b].updated_at || 0;
			return bLastChat - aLastChat;
		});

	// Get visible folders based on showAllFolders state
	$: visibleFolders = showAllFolders ? folderList : folderList.slice(0, MAX_VISIBLE_FOLDERS);
	$: hiddenFolderCount = folderList.length - MAX_VISIBLE_FOLDERS;
	$: hasMoreFolders = folderList.length > MAX_VISIBLE_FOLDERS;

	const onItemMove = (e) => {
		if (e.originFolderId) {
			folderRegistry[e.originFolderId]?.setFolderItems();
		}
	};

	const loadFolderItems = () => {
		for (const folderId of Object.keys(folders)) {
			folderRegistry[folderId]?.setFolderItems();
		}
	};

	$: if (folders || ($selectedFolder && $chatId)) {
		loadFolderItems();
	}
</script>

{#each visibleFolders as folderId (folderId)}
	<RecursiveFolder
		className=""
		bind:folderRegistry
		{folders}
		{folderId}
		{shiftKey}
		{onDelete}
		{onItemMove}
		on:import={(e) => {
			dispatch('import', e.detail);
		}}
		on:update={(e) => {
			dispatch('update', e.detail);
		}}
		on:change={(e) => {
			dispatch('change', e.detail);
		}}
	/>
{/each}

{#if hasMoreFolders}
	<button
		class="w-full flex items-center justify-center gap-1.5 py-1.5 px-2 mt-1 text-xs text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-900 rounded-lg transition"
		on:click={() => {
			showAllFolders = !showAllFolders;
		}}
	>
		{#if showAllFolders}
			<ChevronUp className="size-3" strokeWidth="2.5" />
			<span>{$i18n.t('접기')}</span>
		{:else}
			<ChevronDown className="size-3" strokeWidth="2.5" />
			<span>{$i18n.t('펼치기')} ({hiddenFolderCount})</span>
		{/if}
	</button>
{/if}
