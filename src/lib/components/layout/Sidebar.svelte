<script lang="ts">
	import { toast } from 'svelte-sonner';
	import { v4 as uuidv4 } from 'uuid';

	import { goto } from '$app/navigation';
	import {
		user,
		chats,
		settings,
		showSettings,
		chatId,
		tags,
		folders as _folders,
		showSidebar,
		showSearch,
		mobile,
		showArchivedChats,
		pinnedChats,
		scrollPaginationEnabled,
		currentChatPage,
		temporaryChatEnabled,
		channels,
		socket,
		config,
		isApp,
		models,
		selectedFolder,
		WEBUI_NAME,
		sidebarWidth
	} from '$lib/stores';
	import { onMount, getContext, tick, onDestroy } from 'svelte';

	const i18n = getContext('i18n');

	import {
		getChatList,
		getAllTags,
		getPinnedChatList,
		toggleChatPinnedStatusById,
		getChatById,
		updateChatFolderIdById,
		importChats
	} from '$lib/apis/chats';
	import { createNewFolder, getFolders, updateFolderParentIdById } from '$lib/apis/folders';
	import { WEBUI_API_BASE_URL, WEBUI_BASE_URL } from '$lib/constants';

	import ArchivedChatsModal from './ArchivedChatsModal.svelte';
	import UserMenu from './Sidebar/UserMenu.svelte';
	import ChatItem from './Sidebar/ChatItem.svelte';
	import Spinner from '../common/Spinner.svelte';
	import Loader from '../common/Loader.svelte';
	import Folder from '../common/Folder.svelte';
	import Tooltip from '../common/Tooltip.svelte';
	import Folders from './Sidebar/Folders.svelte';
	import { getChannels, createNewChannel } from '$lib/apis/channels';
	import ChannelModal from './Sidebar/ChannelModal.svelte';
	import ChannelItem from './Sidebar/ChannelItem.svelte';
	import PencilSquare from '../icons/PencilSquare.svelte';
	import Search from '../icons/Search.svelte';
	import SearchModal from './SearchModal.svelte';
	import FolderModal from './Sidebar/Folders/FolderModal.svelte';
	import Sidebar from '../icons/Sidebar.svelte';
	import PinnedModelList from './Sidebar/PinnedModelList.svelte';
	import Note from '../icons/Note.svelte';
	import { slide } from 'svelte/transition';
	import HotkeyHint from '../common/HotkeyHint.svelte';

	const BREAKPOINT = 768;

	let scrollTop = 0;

	let navElement;
	let shiftKey = false;

	let selectedChatId = null;
	let showCreateChannel = false;

	// Pagination variables
	let chatListLoading = false;
	let allChatsLoaded = false;

	let showCreateFolderModal = false;

	let pinnedModels = [];

	let showPinnedModels = false;
	let showChannels = false;
	let showFolders = false;

	let folders = {};
	let folderRegistry = {};

	let newFolderId = null;

	// chatId가 변경되거나, chats 목록이 변경될 때 폴더 목록 갱신
	// (새 대화 시작, 대화 전환, 또는 메시지 전송 시 정렬 반영)
	$: if ($chatId || $selectedFolder || $chats) {
		initFolders();
	}

	const initFolders = async () => {
		if ($config?.features?.enable_folders === false) {
			return;
		}

		const folderList = await getFolders(localStorage.token).catch((error) => {
			return [];
		});
		_folders.set(folderList.sort((a, b) => b.updated_at - a.updated_at));

		folders = {};

		// First pass: Initialize all folder entries
		for (const folder of folderList) {
			// Ensure folder is added to folders with its data
			folders[folder.id] = { ...(folders[folder.id] || {}), ...folder };

			if (newFolderId && folder.id === newFolderId) {
				folders[folder.id].new = true;
				newFolderId = null;
			}
		}

		// Second pass: Tie child folders to their parents
		for (const folder of folderList) {
			if (folder.parent_id) {
				// Ensure the parent folder is initialized if it doesn't exist
				if (!folders[folder.parent_id]) {
					folders[folder.parent_id] = {}; // Create a placeholder if not already present
				}

				// Initialize childrenIds array if it doesn't exist and add the current folder id
				folders[folder.parent_id].childrenIds = folders[folder.parent_id].childrenIds
					? [...folders[folder.parent_id].childrenIds, folder.id]
					: [folder.id];

				// Sort the children by last_chat_updated_at field (or updated_at if no chats)
				folders[folder.parent_id].childrenIds.sort((a, b) => {
					const aLastChat = folders[a].last_chat_updated_at || folders[a].updated_at;
					const bLastChat = folders[b].last_chat_updated_at || folders[b].updated_at;
					return bLastChat - aLastChat;
				});
			}
		}
	};

	const createFolder = async ({ name, data }) => {
		name = name?.trim();
		if (!name) {
			toast.error($i18n.t('Folder name cannot be empty.'));
			return;
		}

		const rootFolders = Object.values(folders).filter((folder) => folder.parent_id === null);
		if (rootFolders.find((folder) => folder.name.toLowerCase() === name.toLowerCase())) {
			// If a folder with the same name already exists, append a number to the name
			let i = 1;
			while (
				rootFolders.find((folder) => folder.name.toLowerCase() === `${name} ${i}`.toLowerCase())
			) {
				i++;
			}

			name = `${name} ${i}`;
		}

		// Add a dummy folder to the list to show the user that the folder is being created
		const tempId = uuidv4();
		folders = {
			...folders,
			tempId: {
				id: tempId,
				name: name,
				created_at: Date.now(),
				updated_at: Date.now()
			}
		};

		const res = await createNewFolder(localStorage.token, {
			name,
			data
		}).catch((error) => {
			toast.error(`${error}`);
			return null;
		});

		if (res) {
			// newFolderId = res.id;
			await initFolders();
			showFolders = true;
		}
	};

	const initChannels = async () => {
		// default (none), group, dm type
		const res = await getChannels(localStorage.token).catch((error) => {
			return null;
		});

		if (res) {
			await channels.set(
				res.sort(
					(a, b) =>
						['', null, 'group', 'dm'].indexOf(a.type) - ['', null, 'group', 'dm'].indexOf(b.type)
				)
			);
		}
	};

	const initChatList = async () => {
		// Reset pagination variables
		console.log('initChatList');
		currentChatPage.set(1);
		allChatsLoaded = false;
		scrollPaginationEnabled.set(false);

		initFolders();
		await Promise.all([
			await (async () => {
				console.log('Init tags');
				const _tags = await getAllTags(localStorage.token);
				tags.set(_tags);
			})(),
			await (async () => {
				console.log('Init pinned chats');
				const _pinnedChats = await getPinnedChatList(localStorage.token);
				pinnedChats.set(_pinnedChats);
			})(),
			await (async () => {
				console.log('Init chat list');
				const _chats = await getChatList(localStorage.token, $currentChatPage);
				await chats.set(_chats);
			})()
		]);

		// Enable pagination
		scrollPaginationEnabled.set(true);
	};

	const loadMoreChats = async () => {
		chatListLoading = true;

		currentChatPage.set($currentChatPage + 1);

		let newChatList = [];

		newChatList = await getChatList(localStorage.token, $currentChatPage);

		// once the bottom of the list has been reached (no results) there is no need to continue querying
		allChatsLoaded = newChatList.length === 0;
		await chats.set([...($chats ? $chats : []), ...newChatList]);

		chatListLoading = false;
	};

	const importChatHandler = async (items, pinned = false, folderId = null) => {
		console.log('importChatHandler', items, pinned, folderId);
		for (const item of items) {
			console.log(item);
			if (item.chat) {
				await importChats(localStorage.token, [
					{
						chat: item.chat,
						meta: item?.meta ?? {},
						pinned: pinned,
						folder_id: folderId,
						created_at: item?.created_at ?? null,
						updated_at: item?.updated_at ?? null
					}
				]);
			}
		}

		initChatList();
	};

	const inputFilesHandler = async (files) => {
		console.log(files);

		for (const file of files) {
			const reader = new FileReader();
			reader.onload = async (e) => {
				const content = e.target.result;

				try {
					const chatItems = JSON.parse(content);
					importChatHandler(chatItems);
				} catch {
					toast.error($i18n.t(`Invalid file format.`));
				}
			};

			reader.readAsText(file);
		}
	};

	const tagEventHandler = async (type, tagName, chatId) => {
		console.log(type, tagName, chatId);
		if (type === 'delete') {
			initChatList();
		} else if (type === 'add') {
			initChatList();
		}
	};

	let draggedOver = false;

	const onDragOver = (e) => {
		e.preventDefault();

		// Check if a file is being draggedOver.
		if (e.dataTransfer?.types?.includes('Files')) {
			draggedOver = true;
		} else {
			draggedOver = false;
		}
	};

	const onDragLeave = () => {
		draggedOver = false;
	};

	const onDrop = async (e) => {
		e.preventDefault();
		console.log(e); // Log the drop event

		// Perform file drop check and handle it accordingly
		if (e.dataTransfer?.files) {
			const inputFiles = Array.from(e.dataTransfer?.files);

			if (inputFiles && inputFiles.length > 0) {
				console.log(inputFiles); // Log the dropped files
				inputFilesHandler(inputFiles); // Handle the dropped files
			}
		}

		draggedOver = false; // Reset draggedOver status after drop
	};

	let touchstart;
	let touchend;

	function checkDirection() {
		const screenWidth = window.innerWidth;
		const swipeDistance = Math.abs(touchend.screenX - touchstart.screenX);
		if (touchstart.clientX < 40 && swipeDistance >= screenWidth / 8) {
			if (touchend.screenX < touchstart.screenX) {
				showSidebar.set(false);
			}
			if (touchend.screenX > touchstart.screenX) {
				showSidebar.set(true);
			}
		}
	}

	const onTouchStart = (e) => {
		touchstart = e.changedTouches[0];
		console.log(touchstart.clientX);
	};

	const onTouchEnd = (e) => {
		touchend = e.changedTouches[0];
		checkDirection();
	};

	const onKeyDown = (e) => {
		if (e.key === 'Shift') {
			shiftKey = true;
		}
	};

	const onKeyUp = (e) => {
		if (e.key === 'Shift') {
			shiftKey = false;
		}
	};

	const onFocus = () => {};

	const onBlur = () => {
		shiftKey = false;
		selectedChatId = null;
	};

	const MIN_WIDTH = 220;
	const MAX_WIDTH = 480;

	let isResizing = false;

	let startWidth = 0;
	let startClientX = 0;

	const resizeStartHandler = (e: MouseEvent) => {
		if ($mobile) return;
		isResizing = true;

		startClientX = e.clientX;
		startWidth = $sidebarWidth ?? 260;

		document.body.style.userSelect = 'none';
	};

	const resizeEndHandler = () => {
		if (!isResizing) return;
		isResizing = false;

		document.body.style.userSelect = '';
		localStorage.setItem('sidebarWidth', String($sidebarWidth));
	};

	const resizeSidebarHandler = (endClientX) => {
		const dx = endClientX - startClientX;
		const newSidebarWidth = Math.min(MAX_WIDTH, Math.max(MIN_WIDTH, startWidth + dx));

		sidebarWidth.set(newSidebarWidth);
		document.documentElement.style.setProperty('--sidebar-width', `${newSidebarWidth}px`);
	};

	let unsubscribers = [];

	onMount(async () => {
		try {
			const width = Number(localStorage.getItem('sidebarWidth'));
			if (!Number.isNaN(width) && width >= MIN_WIDTH && width <= MAX_WIDTH) {
				sidebarWidth.set(width);
			}
		} catch {}

		document.documentElement.style.setProperty('--sidebar-width', `${$sidebarWidth}px`);
		sidebarWidth.subscribe((w) => {
			document.documentElement.style.setProperty('--sidebar-width', `${w}px`);
		});

		await showSidebar.set(!$mobile ? localStorage.sidebar === 'true' : false);

		unsubscribers = [
			mobile.subscribe((value) => {
				if ($showSidebar && value) {
					showSidebar.set(false);
				}

				if ($showSidebar && !value) {
					const navElement = document.getElementsByTagName('nav')[0];
					if (navElement) {
						navElement.style['-webkit-app-region'] = 'drag';
					}
				}
			}),
			showSidebar.subscribe(async (value) => {
				localStorage.sidebar = value;

				// nav element is not available on the first render
				const navElement = document.getElementsByTagName('nav')[0];

				if (navElement) {
					if ($mobile) {
						if (!value) {
							navElement.style['-webkit-app-region'] = 'drag';
						} else {
							navElement.style['-webkit-app-region'] = 'no-drag';
						}
					} else {
						navElement.style['-webkit-app-region'] = 'drag';
					}
				}

				if (value) {
					// Only fetch channels if the feature is enabled and user has permission
					if (
						$config?.features?.enable_channels &&
						($user?.role === 'admin' || ($user?.permissions?.features?.channels ?? true))
					) {
						await initChannels();
					}
					await initChatList();
				}
			}),
			settings.subscribe((value) => {
				if (pinnedModels != value?.pinnedModels ?? []) {
					pinnedModels = value?.pinnedModels ?? [];
					showPinnedModels = pinnedModels.length > 0;
				}
			})
		];

		window.addEventListener('keydown', onKeyDown);
		window.addEventListener('keyup', onKeyUp);

		window.addEventListener('touchstart', onTouchStart);
		window.addEventListener('touchend', onTouchEnd);

		window.addEventListener('focus', onFocus);
		window.addEventListener('blur', onBlur);

		const dropZone = document.getElementById('sidebar');

		dropZone?.addEventListener('dragover', onDragOver);
		dropZone?.addEventListener('drop', onDrop);
		dropZone?.addEventListener('dragleave', onDragLeave);
	});

	onDestroy(() => {
		if (unsubscribers && unsubscribers.length > 0) {
			unsubscribers.forEach((unsubscriber) => {
				if (unsubscriber) {
					unsubscriber();
				}
			});
		}

		window.removeEventListener('keydown', onKeyDown);
		window.removeEventListener('keyup', onKeyUp);

		window.removeEventListener('touchstart', onTouchStart);
		window.removeEventListener('touchend', onTouchEnd);

		window.removeEventListener('focus', onFocus);
		window.removeEventListener('blur', onBlur);

		const dropZone = document.getElementById('sidebar');

		dropZone?.removeEventListener('dragover', onDragOver);
		dropZone?.removeEventListener('drop', onDrop);
		dropZone?.removeEventListener('dragleave', onDragLeave);
	});

	const newChatHandler = async () => {
		selectedChatId = null;
		selectedFolder.set(null);

		if ($user?.role !== 'admin' && $user?.permissions?.chat?.temporary_enforced) {
			await temporaryChatEnabled.set(true);
		} else {
			await temporaryChatEnabled.set(false);
		}

		setTimeout(() => {
			if ($mobile) {
				showSidebar.set(false);
			}
		}, 0);
	};

	const itemClickHandler = async () => {
		selectedChatId = null;
		chatId.set('');

		if ($mobile) {
			showSidebar.set(false);
		}

		await tick();
	};

	const isWindows = /Windows/i.test(navigator.userAgent);
</script>

<ArchivedChatsModal
	bind:show={$showArchivedChats}
	onUpdate={async () => {
		await initChatList();
	}}
/>

<ChannelModal
	bind:show={showCreateChannel}
	onSubmit={async ({ type, name, is_private, access_control, group_ids, user_ids }) => {
		name = name?.trim();

		if (type === 'dm') {
			if (!user_ids || user_ids.length === 0) {
				toast.error($i18n.t('Please select at least one user for Direct Message channel.'));
				return;
			}
		} else {
			if (!name) {
				toast.error($i18n.t('Channel name cannot be empty.'));
				return;
			}
		}

		const res = await createNewChannel(localStorage.token, {
			type: type,
			name: name,
			is_private: is_private,
			access_control: access_control,
			group_ids: group_ids,
			user_ids: user_ids
		}).catch((error) => {
			toast.error(`${error}`);
			return null;
		});

		if (res) {
			$socket.emit('join-channels', { auth: { token: $user?.token } });
			await initChannels();
			showCreateChannel = false;
			showChannels = true;
			goto(`/channels/${res.id}`);
		}
	}}
/>

<FolderModal
	bind:show={showCreateFolderModal}
	onSubmit={async (folder) => {
		await createFolder(folder);
		showCreateFolderModal = false;
	}}
/>

<!-- svelte-ignore a11y-no-static-element-interactions -->

{#if $showSidebar}
	<div
		class=" {$isApp
			? ' ml-[4.5rem] md:ml-0'
			: ''} fixed md:hidden z-40 top-0 right-0 left-0 bottom-0 bg-black/60 w-full min-h-screen h-screen flex justify-center overflow-hidden overscroll-contain"
		on:mousedown={() => {
			showSidebar.set(!$showSidebar);
		}}
	/>
{/if}

<SearchModal
	bind:show={$showSearch}
	onClose={() => {
		if ($mobile) {
			showSidebar.set(false);
		}
	}}
/>

<button
	id="sidebar-new-chat-button"
	class="hidden"
	on:click={() => {
		goto('/');
		newChatHandler();
	}}
/>

<svelte:window
	on:mousemove={(e) => {
		if (!isResizing) return;
		resizeSidebarHandler(e.clientX);
	}}
	on:mouseup={() => {
		resizeEndHandler();
	}}
/>

{#if !$mobile && !$showSidebar}
	<div
		class=" pt-[7px] pb-2 px-2 flex flex-col justify-between text-black dark:text-white hover:bg-gray-50/30 dark:hover:bg-gray-950/30 h-full z-10 transition-all border-e-[0.5px] border-gray-50 dark:border-gray-850/30"
		id="sidebar"
	>
		<button
			class="flex flex-col flex-1 {isWindows ? 'cursor-pointer' : 'cursor-[e-resize]'}"
			on:click={async () => {
				showSidebar.set(!$showSidebar);
			}}
		>
			<div class="pb-1.5">
				<Tooltip
					content={$showSidebar ? $i18n.t('Close Sidebar') : $i18n.t('Open Sidebar')}
					placement="right"
				>
					<button
						class="flex rounded-xl hover:bg-gray-100 dark:hover:bg-gray-850 transition group {isWindows
							? 'cursor-pointer'
							: 'cursor-[e-resize]'}"
						aria-label={$showSidebar ? $i18n.t('Close Sidebar') : $i18n.t('Open Sidebar')}
					>
						<div class=" self-center flex items-center justify-center size-9">
							<img
								src="{WEBUI_BASE_URL}/static/favicon.png"
								class="sidebar-new-chat-icon size-6 rounded-full group-hover:hidden"
								alt=""
							/>

							<Sidebar className="size-5 hidden group-hover:flex" />
						</div>
					</button>
				</Tooltip>
			</div>

			<div class="-mt-[0.5px]">
				<div class="">
					<Tooltip content={$i18n.t('New Chat')} placement="right">
						<a
							class=" cursor-pointer flex rounded-xl hover:bg-gray-100 dark:hover:bg-gray-850 transition group"
							href="/"
							draggable="false"
							on:click={async (e) => {
								e.stopImmediatePropagation();
								e.preventDefault();

								goto('/');
								newChatHandler();
							}}
							aria-label={$i18n.t('New Chat')}
						>
							<div class=" self-center flex items-center justify-center size-9">
								<PencilSquare className="size-4.5" />
							</div>
						</a>
					</Tooltip>
				</div>

				<div>
					<Tooltip content={$i18n.t('Search')} placement="right">
						<button
							class=" cursor-pointer flex rounded-xl hover:bg-gray-100 dark:hover:bg-gray-850 transition group"
							on:click={(e) => {
								e.stopImmediatePropagation();
								e.preventDefault();

								showSearch.set(true);
							}}
							draggable="false"
							aria-label={$i18n.t('Search')}
						>
							<div class=" self-center flex items-center justify-center size-9">
								<Search className="size-4.5" />
							</div>
						</button>
					</Tooltip>
				</div>

				{#if ($config?.features?.enable_notes ?? false) && ($user?.role === 'admin' || ($user?.permissions?.features?.notes ?? true))}
					<div class="">
						<Tooltip content={$i18n.t('Notes')} placement="right">
							<a
								class=" cursor-pointer flex rounded-xl hover:bg-gray-100 dark:hover:bg-gray-850 transition group"
								href="/notes"
								on:click={async (e) => {
									e.stopImmediatePropagation();
									e.preventDefault();

									goto('/notes');
									itemClickHandler();
								}}
								draggable="false"
								aria-label={$i18n.t('Notes')}
							>
								<div class=" self-center flex items-center justify-center size-9">
									<Note className="size-4.5" />
								</div>
							</a>
						</Tooltip>
					</div>
				{/if}

				{#if $user?.role === 'admin' || $user?.permissions?.workspace?.models || $user?.permissions?.workspace?.knowledge || $user?.permissions?.workspace?.prompts || $user?.permissions?.workspace?.tools}
					<div class="">
						<Tooltip content={$i18n.t('Workspace')} placement="right">
							<a
								class=" cursor-pointer flex rounded-xl hover:bg-gray-100 dark:hover:bg-gray-850 transition group"
								href="/workspace"
								on:click={async (e) => {
									e.stopImmediatePropagation();
									e.preventDefault();

									goto('/workspace');
									itemClickHandler();
								}}
								aria-label={$i18n.t('Workspace')}
								draggable="false"
							>
								<div class=" self-center flex items-center justify-center size-9">
									<svg
										xmlns="http://www.w3.org/2000/svg"
										fill="none"
										viewBox="0 0 24 24"
										stroke-width="1.5"
										stroke="currentColor"
										class="size-4.5"
									>
										<path
											stroke-linecap="round"
											stroke-linejoin="round"
											d="M13.5 16.875h3.375m0 0h3.375m-3.375 0V13.5m0 3.375v3.375M6 10.5h2.25a2.25 2.25 0 0 0 2.25-2.25V6a2.25 2.25 0 0 0-2.25-2.25H6A2.25 2.25 0 0 0 3.75 6v2.25A2.25 2.25 0 0 0 6 10.5Zm0 9.75h2.25A2.25 2.25 0 0 0 10.5 18v-2.25a2.25 2.25 0 0 0-2.25-2.25H6a2.25 2.25 0 0 0-2.25 2.25V18A2.25 2.25 0 0 0 6 20.25Zm9.75-9.75H18a2.25 2.25 0 0 0 2.25-2.25V6A2.25 2.25 0 0 0 18 3.75h-2.25A2.25 2.25 0 0 0 13.5 6v2.25a2.25 2.25 0 0 0 2.25 2.25Z"
										/>
									</svg>
								</div>
							</a>
						</Tooltip>
					</div>
				{/if}
			</div>
		</button>

		<div>
			<div>
				<div class=" py-2 flex justify-center items-center">
					{#if $user !== undefined && $user !== null}
						<UserMenu
							role={$user?.role}
							profile={$config?.features?.enable_user_status ?? true}
							showActiveUsers={false}
							on:show={(e) => {
								if (e.detail === 'archived-chat') {
									showArchivedChats.set(true);
								}
							}}
						>
							<div
								class=" cursor-pointer flex rounded-xl hover:bg-gray-100 dark:hover:bg-gray-850 transition group"
							>
								<div class="self-center relative">
									<img
										src={`${WEBUI_API_BASE_URL}/users/${$user?.id}/profile/image`}
										class=" size-7 object-cover rounded-full"
										alt={$i18n.t('Open User Profile Menu')}
										aria-label={$i18n.t('Open User Profile Menu')}
									/>

									{#if $config?.features?.enable_user_status}
										<div class="absolute -bottom-0.5 -right-0.5">
											<span class="relative flex size-2.5">
												<span
													class="absolute inline-flex h-full w-full animate-ping rounded-full bg-green-400 opacity-75"
												></span>
												<span
													class="relative inline-flex size-2.5 rounded-full {true
														? 'bg-green-500'
														: 'bg-gray-300 dark:bg-gray-700'} border-2 border-white dark:border-gray-900"
												></span>
											</span>
										</div>
									{/if}
								</div>
							</div>
						</UserMenu>
					{/if}
				</div>
			</div>
		</div>
	</div>
{/if}

<!-- {$i18n.t('New Folder')} -->
<!-- {$i18n.t('Pinned')} -->

{#if $showSidebar}
	<div
		bind:this={navElement}
		id="sidebar"
		class="h-screen max-h-[100dvh] min-h-screen select-none {$showSidebar
			? `${$mobile ? 'bg-gray-50 dark:bg-gray-950' : 'bg-gray-50/70 dark:bg-gray-950/70'} z-50`
			: ' bg-transparent z-0 '} {$isApp
			? `ml-[4.5rem] md:ml-0 `
			: ' transition-all duration-300 '} shrink-0 text-gray-900 dark:text-gray-200 text-sm fixed top-0 left-0 overflow-x-hidden
        "
		transition:slide={{ duration: 250, axis: 'x' }}
		data-state={$showSidebar}
	>
		<div
			class=" my-auto flex flex-col justify-between h-screen max-h-[100dvh] w-[var(--sidebar-width)] overflow-x-hidden scrollbar-hidden z-50 {$showSidebar
				? ''
				: 'invisible'}"
		>
			<div
				class="sidebar px-[0.5625rem] pt-2 pb-1.5 flex justify-between space-x-1 text-gray-600 dark:text-gray-400 sticky top-0 z-10 -mb-3"
			>
				<a
					class="flex items-center rounded-xl size-8.5 h-full justify-center hover:bg-gray-100/50 dark:hover:bg-gray-850/50 transition no-drag-region"
					href="/"
					draggable="false"
					on:click={newChatHandler}
				>
					<img
						crossorigin="anonymous"
						src="{WEBUI_BASE_URL}/static/favicon.png"
						class="sidebar-new-chat-icon size-6 rounded-full"
						alt=""
					/>
				</a>

				<a href="/" class="flex flex-1 px-1.5" on:click={newChatHandler}>
					<div
						id="sidebar-webui-name"
						class=" self-center font-medium text-gray-850 dark:text-white font-primary"
					>
						{$WEBUI_NAME}
					</div>
				</a>
				<Tooltip
					content={$showSidebar ? $i18n.t('Close Sidebar') : $i18n.t('Open Sidebar')}
					placement="bottom"
				>
					<button
						class="flex rounded-xl size-8.5 justify-center items-center hover:bg-gray-100/50 dark:hover:bg-gray-850/50 transition {isWindows
							? 'cursor-pointer'
							: 'cursor-[w-resize]'}"
						on:click={() => {
							showSidebar.set(!$showSidebar);
						}}
						aria-label={$showSidebar ? $i18n.t('Close Sidebar') : $i18n.t('Open Sidebar')}
					>
						<div class=" self-center p-1.5">
							<Sidebar />
						</div>
					</button>
				</Tooltip>

				<div
					class="{scrollTop > 0
						? 'visible'
						: 'invisible'} sidebar-bg-gradient-to-b bg-linear-to-b from-gray-50 dark:from-gray-950 to-transparent from-50% pointer-events-none absolute inset-0 -z-10 -mb-6"
				></div>
			</div>

			<div
				class="relative flex flex-col flex-1 overflow-y-auto scrollbar-hidden pt-3 pb-3"
				on:scroll={(e) => {
					if (e.target.scrollTop === 0) {
						scrollTop = 0;
					} else {
						scrollTop = e.target.scrollTop;
					}
				}}
			>
				<div class="pb-1.5">
					<div class="px-[0.4375rem] flex justify-center text-gray-800 dark:text-gray-200">
						<a
							id="sidebar-new-chat-button"
							class="group grow flex items-center space-x-3 rounded-2xl px-2.5 py-2 hover:bg-gray-100 dark:hover:bg-gray-900 transition outline-none"
							href="/"
							draggable="false"
							on:click={newChatHandler}
							aria-label={$i18n.t('New Chat')}
						>
							<div class="self-center">
								<PencilSquare className=" size-4.5" strokeWidth="2" />
							</div>

							<div class="flex flex-1 self-center translate-y-[0.5px]">
								<div class=" self-center text-sm font-primary">{$i18n.t('New Chat')}</div>
							</div>

							<HotkeyHint name="newChat" className=" group-hover:visible invisible" />
						</a>
					</div>

					<div class="px-[0.4375rem] flex justify-center text-gray-800 dark:text-gray-200">
						<button
							id="sidebar-search-button"
							class="group grow flex items-center space-x-3 rounded-2xl px-2.5 py-2 hover:bg-gray-100 dark:hover:bg-gray-900 transition outline-none"
							on:click={() => {
								showSearch.set(true);
							}}
							draggable="false"
							aria-label={$i18n.t('Search')}
						>
							<div class="self-center">
								<Search strokeWidth="2" className="size-4.5" />
							</div>

							<div class="flex flex-1 self-center translate-y-[0.5px]">
								<div class=" self-center text-sm font-primary">{$i18n.t('Search')}</div>
							</div>
							<HotkeyHint name="search" className=" group-hover:visible invisible" />
						</button>
					</div>

					{#if ($config?.features?.enable_notes ?? false) && ($user?.role === 'admin' || ($user?.permissions?.features?.notes ?? true))}
						<div class="px-[0.4375rem] flex justify-center text-gray-800 dark:text-gray-200">
							<a
								id="sidebar-notes-button"
								class="grow flex items-center space-x-3 rounded-2xl px-2.5 py-2 hover:bg-gray-100 dark:hover:bg-gray-900 transition"
								href="/notes"
								on:click={itemClickHandler}
								draggable="false"
								aria-label={$i18n.t('Notes')}
							>
								<div class="self-center">
									<Note className="size-4.5" strokeWidth="2" />
								</div>

								<div class="flex self-center translate-y-[0.5px]">
									<div class=" self-center text-sm font-primary">{$i18n.t('Notes')}</div>
								</div>
							</a>
						</div>
					{/if}

					{#if $user?.role === 'admin' || $user?.permissions?.workspace?.models || $user?.permissions?.workspace?.knowledge || $user?.permissions?.workspace?.prompts || $user?.permissions?.workspace?.tools}
						<div class="px-[0.4375rem] flex justify-center text-gray-800 dark:text-gray-200">
							<a
								id="sidebar-workspace-button"
								class="grow flex items-center space-x-3 rounded-2xl px-2.5 py-2 hover:bg-gray-100 dark:hover:bg-gray-900 transition"
								href="/workspace"
								on:click={itemClickHandler}
								draggable="false"
								aria-label={$i18n.t('Workspace')}
							>
								<div class="self-center">
									<svg
										xmlns="http://www.w3.org/2000/svg"
										fill="none"
										viewBox="0 0 24 24"
										stroke-width="2"
										stroke="currentColor"
										class="size-4.5"
									>
										<path
											stroke-linecap="round"
											stroke-linejoin="round"
											d="M13.5 16.875h3.375m0 0h3.375m-3.375 0V13.5m0 3.375v3.375M6 10.5h2.25a2.25 2.25 0 0 0 2.25-2.25V6a2.25 2.25 0 0 0-2.25-2.25H6A2.25 2.25 0 0 0 3.75 6v2.25A2.25 2.25 0 0 0 6 10.5Zm0 9.75h2.25A2.25 2.25 0 0 0 10.5 18v-2.25a2.25 2.25 0 0 0-2.25-2.25H6a2.25 2.25 0 0 0-2.25 2.25V18A2.25 2.25 0 0 0 6 20.25Zm9.75-9.75H18a2.25 2.25 0 0 0 2.25-2.25V6A2.25 2.25 0 0 0 18 3.75h-2.25A2.25 2.25 0 0 0 13.5 6v2.25a2.25 2.25 0 0 0 2.25 2.25Z"
										/>
									</svg>
								</div>

								<div class="flex self-center translate-y-[0.5px]">
									<div class=" self-center text-sm font-primary">{$i18n.t('Workspace')}</div>
								</div>
							</a>
						</div>
					{/if}
				</div>

				{#if ($models ?? []).length > 0 && (($settings?.pinnedModels ?? []).length > 0 || $config?.default_pinned_models)}
					<Folder
						id="sidebar-models"
						bind:open={showPinnedModels}
						className="px-2 mt-0.5"
						name={$i18n.t('Models')}
						chevron={false}
						dragAndDrop={false}
					>
						<PinnedModelList bind:selectedChatId {shiftKey} />
					</Folder>
				{/if}

				{#if $config?.features?.enable_channels && ($user?.role === 'admin' || ($user?.permissions?.features?.channels ?? true))}
					<Folder
						id="sidebar-channels"
						bind:open={showChannels}
						className="px-2 mt-0.5"
						name={$i18n.t('Channels')}
						chevron={false}
						dragAndDrop={false}
						onAdd={$user?.role === 'admin' || ($user?.permissions?.features?.channels ?? true)
							? async () => {
									await tick();

									setTimeout(() => {
										showCreateChannel = true;
									}, 0);
								}
							: null}
						onAddLabel={$i18n.t('Create Channel')}
					>
						{#each $channels as channel, channelIdx (`${channel?.id}`)}
							<ChannelItem
								{channel}
								onUpdate={async () => {
									await initChannels();
								}}
							/>

							{#if channelIdx < $channels.length - 1 && channel.type !== $channels[channelIdx + 1]?.type}<hr
									class=" border-gray-100/40 dark:border-gray-800/10 my-1.5 w-full"
								/>
							{/if}
						{/each}
					</Folder>
				{/if}

				{#if $config?.features?.enable_folders && ($user?.role === 'admin' || ($user?.permissions?.features?.folders ?? true))}
					<Folder
						id="sidebar-folders"
						bind:open={showFolders}
						className="px-2 mt-0.5"
						name={$i18n.t('Folders')}
						chevron={false}
						onAdd={() => {
							showCreateFolderModal = true;
						}}
						onAddLabel={$i18n.t('New Folder')}
						on:drop={async (e) => {
							const { type, id, item } = e.detail;

							if (type === 'folder') {
								if (folders[id].parent_id === null) {
									return;
								}

								const res = await updateFolderParentIdById(localStorage.token, id, null).catch(
									(error) => {
										toast.error(`${error}`);
										return null;
									}
								);

								if (res) {
									await initFolders();
								}
							}
						}}
					>
						<!-- 새폴더 생성 버튼 -->
						<div class="px-2 mt-1">
							<button
								class="w-full flex items-center space-x-2 rounded-lg px-2 py-2 hover:bg-gray-100 dark:hover:bg-gray-900 transition text-sm text-gray-900 dark:text-gray-100"
								on:click={() => {
									showCreateFolderModal = true;
								}}
							>
								<div class="self-center">
									<svg
										width="20"
										height="20"
										viewBox="0 0 20 20"
										fill="currentColor"
										xmlns="http://www.w3.org/2000/svg"
										class="icon"
										aria-hidden="true"
										><path
											d="M15.2041 17.5V15.665H13.3691C13.0019 15.665 12.7041 15.3673 12.7041 15C12.7041 14.6327 13.0019 14.335 13.3691 14.335H15.2041V12.5C15.2041 12.1327 15.5019 11.835 15.8691 11.835C16.2362 11.8352 16.5332 12.1329 16.5332 12.5V14.335H18.3691C18.7362 14.3352 19.0332 14.6329 19.0332 15C19.0332 15.3671 18.7362 15.6648 18.3691 15.665H16.5332V17.5C16.5332 17.8671 16.2362 18.1648 15.8691 18.165C15.5019 18.165 15.2041 17.8673 15.2041 17.5ZM2.12012 12.7002V7.29981C2.12012 6.64581 2.11922 6.1149 2.1543 5.68555C2.19002 5.24867 2.26619 4.85832 2.45117 4.49512L2.56836 4.28516C2.86045 3.80898 3.27979 3.42103 3.78028 3.16602L3.91797 3.10156C4.24192 2.96268 4.5885 2.90039 4.97071 2.86914C5.40006 2.83406 5.93096 2.83496 6.58496 2.83496H7.28028C7.42346 2.83496 7.52305 2.83479 7.6211 2.84082L7.875 2.86719C8.46133 2.95309 9.01189 3.20874 9.45703 3.60547L9.70215 3.84473C9.81425 3.95779 9.85105 3.99455 9.88672 4.02637L9.99805 4.11719C10.2646 4.31741 10.5851 4.43638 10.9199 4.45703L11.1797 4.45996H13.6914C14.2499 4.45996 14.703 4.45958 15.0713 4.48535C15.4458 4.51157 15.7828 4.56683 16.1025 4.70313L16.3662 4.83106C16.9638 5.15706 17.4378 5.67623 17.707 6.30762L17.7939 6.54981C17.868 6.79538 17.904 7.05317 17.9238 7.33203C17.9498 7.69789 17.9502 8.14747 17.9502 8.7002C17.9501 8.87631 17.8803 9.0453 17.7559 9.16992C17.6311 9.29464 17.4615 9.36524 17.2852 9.36524H3.4502V12.7002C3.4502 13.3761 3.45084 13.8434 3.48047 14.2061C3.50947 14.5608 3.56304 14.7568 3.63672 14.9014L3.70215 15.0195C3.86642 15.2873 4.10236 15.505 4.38379 15.6484L4.50391 15.7002C4.63661 15.7476 4.81329 15.783 5.0791 15.8047C5.44174 15.8343 5.90903 15.835 6.58496 15.835H9.40918L9.54395 15.8486C9.84681 15.9108 10.0742 16.1788 10.0742 16.5C10.0742 16.8212 9.84681 17.0892 9.54395 17.1514L9.40918 17.165H6.58496C5.93096 17.165 5.40006 17.1659 4.97071 17.1309C4.5885 17.0996 4.24192 17.0373 3.91797 16.8984L3.78028 16.834C3.27979 16.579 2.86045 16.191 2.56836 15.7148L2.45117 15.5049C2.26619 15.1417 2.19002 14.7513 2.1543 14.3145C2.11922 13.8851 2.12012 13.3542 2.12012 12.7002ZM3.4502 8.03516H16.6172C16.6146 7.79548 16.6098 7.59777 16.5977 7.42676C16.5816 7.20054 16.5552 7.04845 16.5205 6.9336L16.4834 6.8291C16.332 6.47411 16.0655 6.1824 15.7295 5.99903L15.5811 5.92676C15.4545 5.8728 15.2835 5.83385 14.9785 5.8125C14.6674 5.79073 14.2686 5.79004 13.6914 5.79004H11.1797L10.8379 5.78418C10.2426 5.74746 9.67313 5.53663 9.19922 5.18067L9.00196 5.01953C8.92848 4.95403 8.85889 4.88222 8.75781 4.78028L8.57227 4.59863C8.32169 4.37525 8.01175 4.23086 7.68164 4.18262L7.54004 4.16797C7.49225 4.16502 7.43987 4.16504 7.28028 4.16504H6.58496C5.90903 4.16504 5.44174 4.16569 5.0791 4.19531C4.81329 4.21705 4.63661 4.25237 4.50391 4.29981L4.38379 4.35156C4.10236 4.49499 3.86642 4.71271 3.70215 4.98047L3.63672 5.09863C3.56304 5.24324 3.50947 5.43924 3.48047 5.79395C3.45084 6.15659 3.4502 6.62388 3.4502 7.29981V8.03516Z"
										></path></svg
									>
								</div>
								<div class="self-center">
									{$i18n.t('Create New Folder')}
								</div>
							</button>
						</div>
						<Folders
							bind:folderRegistry
							{folders}
							{shiftKey}
							onDelete={(folderId) => {
								selectedFolder.set(null);
								initChatList();
							}}
							on:update={() => {
								initChatList();
							}}
							on:import={(e) => {
								const { folderId, items } = e.detail;
								importChatHandler(items, false, folderId);
							}}
							on:change={async () => {
								initChatList();
							}}
						/>
					</Folder>
				{/if}

				<Folder
					id="sidebar-chats"
					className="px-2 mt-0.5"
					name={$i18n.t('Chats')}
					chevron={false}
					on:change={async (e) => {
						selectedFolder.set(null);
					}}
					on:import={(e) => {
						importChatHandler(e.detail);
					}}
					on:drop={async (e) => {
						const { type, id, item } = e.detail;

						if (type === 'chat') {
							let chat = await getChatById(localStorage.token, id).catch((error) => {
								return null;
							});
							if (!chat && item) {
								chat = await importChats(localStorage.token, [
									{
										chat: item.chat,
										meta: item?.meta ?? {},
										pinned: false,
										folder_id: null,
										created_at: item?.created_at ?? null,
										updated_at: item?.updated_at ?? null
									}
								]);
							}

							if (chat) {
								console.log(chat);
								if (chat.folder_id) {
									const res = await updateChatFolderIdById(localStorage.token, chat.id, null).catch(
										(error) => {
											toast.error(`${error}`);
											return null;
										}
									);

									folderRegistry[chat.folder_id]?.setFolderItems();
								}

								if (chat.pinned) {
									const res = await toggleChatPinnedStatusById(localStorage.token, chat.id);
								}

								initChatList();
							}
						} else if (type === 'folder') {
							if (folders[id].parent_id === null) {
								return;
							}

							const res = await updateFolderParentIdById(localStorage.token, id, null).catch(
								(error) => {
									toast.error(`${error}`);
									return null;
								}
							);

							if (res) {
								await initFolders();
							}
						}
					}}
				>
					{#if $pinnedChats.length > 0}
						<div class="mb-1">
							<div class="flex flex-col space-y-1 rounded-xl">
								<Folder
									id="sidebar-pinned-chats"
									buttonClassName=" text-gray-500"
									on:import={(e) => {
										importChatHandler(e.detail, true);
									}}
									on:drop={async (e) => {
										const { type, id, item } = e.detail;

										if (type === 'chat') {
											let chat = await getChatById(localStorage.token, id).catch((error) => {
												return null;
											});
											if (!chat && item) {
												chat = await importChats(localStorage.token, [
													{
														chat: item.chat,
														meta: item?.meta ?? {},
														pinned: false,
														folder_id: null,
														created_at: item?.created_at ?? null,
														updated_at: item?.updated_at ?? null
													}
												]);
											}

											if (chat) {
												console.log(chat);
												if (chat.folder_id) {
													const res = await updateChatFolderIdById(
														localStorage.token,
														chat.id,
														null
													).catch((error) => {
														toast.error(`${error}`);
														return null;
													});
												}

												if (!chat.pinned) {
													const res = await toggleChatPinnedStatusById(localStorage.token, chat.id);
												}

												initChatList();
											}
										}
									}}
									name={$i18n.t('Pinned')}
								>
									<div
										class="ml-3 pl-1 mt-[1px] flex flex-col overflow-y-auto scrollbar-hidden border-s border-gray-100 dark:border-gray-900 text-gray-900 dark:text-gray-200"
									>
										{#each $pinnedChats as chat, idx (`pinned-chat-${chat?.id ?? idx}`)}
											<ChatItem
												className=""
												id={chat.id}
												title={chat.title}
												{shiftKey}
												selected={selectedChatId === chat.id}
												on:select={() => {
													selectedChatId = chat.id;
												}}
												on:unselect={() => {
													selectedChatId = null;
												}}
												on:change={async () => {
													initChatList();
												}}
												on:tag={(e) => {
													const { type, name } = e.detail;
													tagEventHandler(type, name, chat.id);
												}}
											/>
										{/each}
									</div>
								</Folder>
							</div>
						</div>
					{/if}

					<div class=" flex-1 flex flex-col overflow-y-auto scrollbar-hidden">
						<div class="pt-1.5">
							{#if $chats}
								{#each $chats as chat, idx (`chat-${chat?.id ?? idx}`)}
									{#if idx === 0 || (idx > 0 && chat.time_range !== $chats[idx - 1].time_range)}
										<div
											class="w-full pl-2.5 text-xs text-gray-500 dark:text-gray-500 font-medium {idx ===
											0
												? ''
												: 'pt-5'} pb-1.5"
										>
											{$i18n.t(chat.time_range)}
											<!-- localisation keys for time_range to be recognized from the i18next parser (so they don't get automatically removed):
							{$i18n.t('Today')}
							{$i18n.t('Yesterday')}
							{$i18n.t('Previous 7 days')}
							{$i18n.t('Previous 30 days')}
							{$i18n.t('January')}
							{$i18n.t('February')}
							{$i18n.t('March')}
							{$i18n.t('April')}
							{$i18n.t('May')}
							{$i18n.t('June')}
							{$i18n.t('July')}
							{$i18n.t('August')}
							{$i18n.t('September')}
							{$i18n.t('October')}
							{$i18n.t('November')}
							{$i18n.t('December')}
							-->
										</div>
									{/if}

									<ChatItem
										className=""
										id={chat.id}
										title={chat.title}
										{shiftKey}
										selected={selectedChatId === chat.id}
										on:select={() => {
											selectedChatId = chat.id;
										}}
										on:unselect={() => {
											selectedChatId = null;
										}}
										on:change={async () => {
											initChatList();
										}}
										on:tag={(e) => {
											const { type, name } = e.detail;
											tagEventHandler(type, name, chat.id);
										}}
									/>
								{/each}

								{#if $scrollPaginationEnabled && !allChatsLoaded}
									<Loader
										on:visible={(e) => {
											if (!chatListLoading) {
												loadMoreChats();
											}
										}}
									>
										<div
											class="w-full flex justify-center py-1 text-xs animate-pulse items-center gap-2"
										>
											<Spinner className=" size-4" />
											<div class=" ">{$i18n.t('Loading...')}</div>
										</div>
									</Loader>
								{/if}
							{:else}
								<div
									class="w-full flex justify-center py-1 text-xs animate-pulse items-center gap-2"
								>
									<Spinner className=" size-4" />
									<div class=" ">{$i18n.t('Loading...')}</div>
								</div>
							{/if}
						</div>
					</div>
				</Folder>
			</div>

			<div class="px-1.5 pt-1.5 pb-2 sticky bottom-0 z-10 -mt-3 sidebar">
				<div
					class=" sidebar-bg-gradient-to-t bg-linear-to-t from-gray-50 dark:from-gray-950 to-transparent from-50% pointer-events-none absolute inset-0 -z-10 -mt-6"
				></div>
				<div class="flex flex-col font-primary">
					{#if $user !== undefined && $user !== null}
						<UserMenu
							role={$user?.role}
							profile={$config?.features?.enable_user_status ?? true}
							showActiveUsers={false}
							on:show={(e) => {
								if (e.detail === 'archived-chat') {
									showArchivedChats.set(true);
								}
							}}
						>
							<div
								class=" flex items-center rounded-2xl py-2 px-1.5 w-full hover:bg-gray-100/50 dark:hover:bg-gray-900/50 transition"
							>
								<div class=" self-center mr-3 relative">
									<img
										src={`${WEBUI_API_BASE_URL}/users/${$user?.id}/profile/image`}
										class=" size-7 object-cover rounded-full"
										alt={$i18n.t('Open User Profile Menu')}
										aria-label={$i18n.t('Open User Profile Menu')}
									/>

									{#if $config?.features?.enable_user_status}
										<div class="absolute -bottom-0.5 -right-0.5">
											<span class="relative flex size-2.5">
												<span
													class="absolute inline-flex h-full w-full animate-ping rounded-full bg-green-400 opacity-75"
												></span>
												<span
													class="relative inline-flex size-2.5 rounded-full {true
														? 'bg-green-500'
														: 'bg-gray-300 dark:bg-gray-700'} border-2 border-white dark:border-gray-900"
												></span>
											</span>
										</div>
									{/if}
								</div>
								<div class=" self-center font-medium">{$user?.name}</div>
							</div>
						</UserMenu>
					{/if}
				</div>
			</div>
		</div>
	</div>

	{#if !$mobile}
		<div
			class="relative flex items-center justify-center group border-l border-gray-50 dark:border-gray-850/30 hover:border-gray-200 dark:hover:border-gray-800 transition z-20"
			id="sidebar-resizer"
			on:mousedown={resizeStartHandler}
			role="separator"
		>
			<div
				class=" absolute -left-1.5 -right-1.5 -top-0 -bottom-0 z-20 cursor-col-resize bg-transparent"
			/>
		</div>
	{/if}
{/if}
