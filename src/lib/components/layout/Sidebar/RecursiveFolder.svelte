<script>
	import { getContext, createEventDispatcher, onMount, onDestroy, tick } from 'svelte';

	const i18n = getContext('i18n');
	const dispatch = createEventDispatcher();

	import DOMPurify from 'dompurify';
	import fileSaver from 'file-saver';
	const { saveAs } = fileSaver;

	import { goto } from '$app/navigation';
	import { toast } from 'svelte-sonner';

	import { chatId, mobile, selectedFolder, showSidebar } from '$lib/stores';

	import {
		deleteFolderById,
		updateFolderIsExpandedById,
		updateFolderById,
		updateFolderParentIdById,
		getFolderById
	} from '$lib/apis/folders';
	import {
		getChatById,
		getChatsByFolderId,
		getChatListByFolderId,
		updateChatFolderIdById,
		importChats
	} from '$lib/apis/chats';

	import ChevronDown from '../../icons/ChevronDown.svelte';
	import ChevronRight from '../../icons/ChevronRight.svelte';
	import Collapsible from '../../common/Collapsible.svelte';
	import DragGhost from '$lib/components/common/DragGhost.svelte';

	import FolderOpen from '$lib/components/icons/FolderOpen.svelte';
	import EllipsisHorizontal from '$lib/components/icons/EllipsisHorizontal.svelte';

	import ChatItem from './ChatItem.svelte';
	import FolderMenu from './Folders/FolderMenu.svelte';
	import DeleteConfirmDialog from '$lib/components/common/ConfirmDialog.svelte';
	import FolderModal from './Folders/FolderModal.svelte';
	import Emoji from '$lib/components/common/Emoji.svelte';
	import Spinner from '$lib/components/common/Spinner.svelte';

	export let folderRegistry = {};
	export let open = false;

	export let folders;
	export let folderId;
	export let shiftKey = false;

	export let className = '';

	export let deleteFolderContents = true;

	export let parentDragged = false;

	export let onDelete = (e) => {};
	export let onItemMove = (e) => {};

	let folderElement;

	let showFolderModal = false;
	let edit = false;

	let draggedOver = false;
	let dragged = false;

	let clickTimer = null;

	let name = '';

	const onDragOver = (e) => {
		e.preventDefault();
		e.stopPropagation();
		if (dragged || parentDragged) {
			return;
		}
		draggedOver = true;
	};

	const onDrop = async (e) => {
		e.preventDefault();
		e.stopPropagation();
		if (dragged || parentDragged) {
			return;
		}

		if (folderElement.contains(e.target)) {
			console.log('Dropped on the Button');

			if (e.dataTransfer.items && e.dataTransfer.items.length > 0) {
				// Iterate over all items in the DataTransferItemList use functional programming
				for (const item of Array.from(e.dataTransfer.items)) {
					// If dropped items aren't files, reject them
					if (item.kind === 'file') {
						const file = item.getAsFile();
						if (file && file.type === 'application/json') {
							console.log('Dropped file is a JSON file!');

							// Read the JSON file with FileReader
							const reader = new FileReader();
							reader.onload = async function (event) {
								try {
									const fileContent = JSON.parse(event.target.result);
									open = true;
									dispatch('import', {
										folderId: folderId,
										items: fileContent
									});
								} catch (error) {
									console.error('Error parsing JSON file:', error);
								}
							};

							// Start reading the file
							reader.readAsText(file);
						} else {
							console.error('Only JSON file types are supported.');
						}

						console.log(file);
					} else {
						// Handle the drag-and-drop data for folders or chats (same as before)
						const dataTransfer = e.dataTransfer.getData('text/plain');

						try {
							const data = JSON.parse(dataTransfer);
							console.log(data);

							const { type, id, item } = data;

							if (type === 'folder') {
								open = true;
								if (id === folderId) {
									return;
								}
								// Move the folder
								const res = await updateFolderParentIdById(localStorage.token, id, folderId).catch(
									(error) => {
										toast.error(`${error}`);
										return null;
									}
								);

								if (res) {
									dispatch('update');
								}
							} else if (type === 'chat') {
								open = true;

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
									]).catch((error) => {
										toast.error(`${error}`);
										return null;
									});
								}

								// Move the chat
								const res = await updateChatFolderIdById(
									localStorage.token,
									chat.id,
									folderId
								).catch((error) => {
									toast.error(`${error}`);
									return null;
								});

								onItemMove({
									originFolderId: chat.folder_id,
									targetFolderId: folderId,
									e
								});

								if (res) {
									dispatch('update');
								}
							}
						} catch (error) {
							console.log('Error parsing dataTransfer:', error);
						}
					}
				}
			}

			setFolderItems();
			draggedOver = false;
		}
	};

	const onDragLeave = (e) => {
		e.preventDefault();
		if (dragged || parentDragged) {
			return;
		}

		draggedOver = false;
	};

	const dragImage = new Image();
	dragImage.src =
		'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=';

	let x;
	let y;

	const onDragStart = (event) => {
		event.stopPropagation();
		event.dataTransfer.setDragImage(dragImage, 0, 0);

		// Set the data to be transferred
		event.dataTransfer.setData(
			'text/plain',
			JSON.stringify({
				type: 'folder',
				id: folderId
			})
		);

		dragged = true;
		folderElement.style.opacity = '0.5'; // Optional: Visual cue to show it's being dragged
	};

	const onDrag = (event) => {
		event.stopPropagation();

		x = event.clientX;
		y = event.clientY;
	};

	const onDragEnd = (event) => {
		event.stopPropagation();

		folderElement.style.opacity = '1'; // Reset visual cue after drag
		dragged = false;
	};

	onMount(async () => {
		open = folders[folderId].is_expanded;
		folderRegistry[folderId] = {
			setFolderItems: () => {
				setFolderItems();
			}
		};
		if (folderElement) {
			folderElement.addEventListener('dragover', onDragOver);
			folderElement.addEventListener('drop', onDrop);
			folderElement.addEventListener('dragleave', onDragLeave);

			// Event listener for when dragging starts
			folderElement.addEventListener('dragstart', onDragStart);
			// Event listener for when dragging occurs (optional)
			folderElement.addEventListener('drag', onDrag);
			// Event listener for when dragging ends
			folderElement.addEventListener('dragend', onDragEnd);
		}

		if (folders[folderId]?.new) {
			delete folders[folderId].new;
			await tick();
			renameHandler();
		}
	});

	onDestroy(() => {
		if (folderElement) {
			folderElement.addEventListener('dragover', onDragOver);
			folderElement.removeEventListener('drop', onDrop);
			folderElement.removeEventListener('dragleave', onDragLeave);

			folderElement.removeEventListener('dragstart', onDragStart);
			folderElement.removeEventListener('drag', onDrag);
			folderElement.removeEventListener('dragend', onDragEnd);
		}
	});

	let showDeleteConfirm = false;

	const deleteHandler = async () => {
		const res = await deleteFolderById(localStorage.token, folderId, deleteFolderContents).catch(
			(error) => {
				toast.error(`${error}`);
				return null;
			}
		);

		if (res) {
			toast.success($i18n.t('Folder deleted successfully'));
			onDelete(folderId);
		}
	};

	const updateHandler = async ({ name, meta, data }) => {
		if (name === '') {
			toast.error($i18n.t('Folder name cannot be empty.'));
			return;
		}

		const currentName = folders[folderId].name;

		name = name.trim();
		folders[folderId].name = name;

		const res = await updateFolderById(localStorage.token, folderId, {
			name,
			...(meta ? { meta } : {}),
			...(data ? { data } : {})
		}).catch((error) => {
			toast.error(`${error}`);

			folders[folderId].name = currentName;
			return null;
		});

		if (res) {
			folders[folderId].name = name;
			if (data) {
				folders[folderId].data = data;
			}

			// toast.success($i18n.t('Folder name updated successfully'));
			toast.success($i18n.t('Folder updated successfully'));

			if ($selectedFolder?.id === folderId) {
				const folder = await getFolderById(localStorage.token, folderId).catch((error) => {
					toast.error(`${error}`);
					return null;
				});

				if (folder) {
					await selectedFolder.set(folder);
				}
			}
			dispatch('update');
		}
	};

	const isExpandedUpdateHandler = async () => {
		const res = await updateFolderIsExpandedById(localStorage.token, folderId, open).catch(
			(error) => {
				toast.error(`${error}`);
				return null;
			}
		);
	};

	let isExpandedUpdateTimeout;

	const isExpandedUpdateDebounceHandler = () => {
		clearTimeout(isExpandedUpdateTimeout);
		isExpandedUpdateTimeout = setTimeout(() => {
			isExpandedUpdateHandler();
		}, 500);
	};

	let chats = null;
	export const setFolderItems = async () => {
		await tick();
		if (open) {
			chats = await getChatListByFolderId(localStorage.token, folderId).catch((error) => {
				toast.error(`${error}`);
				return [];
			});
		} else {
			chats = null;
		}
	};

	$: if (open) {
		setFolderItems();
	}

	const renameHandler = async () => {
		console.log('Edit');
		await tick();
		name = folders[folderId].name;
		edit = true;

		await tick();
		await tick();

		const input = document.getElementById(`folder-${folderId}-input`);
		if (input) {
			input.focus();
			input.select();
		}
	};

	const exportHandler = async () => {
		const chats = await getChatsByFolderId(localStorage.token, folderId).catch((error) => {
			toast.error(`${error}`);
			return null;
		});
		if (!chats) {
			return;
		}

		const blob = new Blob([JSON.stringify(chats)], {
			type: 'application/json'
		});

		saveAs(blob, `folder-${folders[folderId].name}-export-${Date.now()}.json`);
	};
</script>

<DeleteConfirmDialog
	bind:show={showDeleteConfirm}
	title={$i18n.t('Delete folder?')}
	on:confirm={() => {
		deleteHandler();
	}}
>
	<div class=" text-sm text-gray-700 dark:text-gray-300 flex-1 line-clamp-3 mb-2">
		<!-- {$i18n.t('This will delete <strong>{{NAME}}</strong> and <strong>all its contents</strong>.', {
				NAME: folders[folderId].name
			})} -->

		{$i18n.t(`Are you sure you want to delete "{{NAME}}"?`, {
			NAME: folders[folderId].name
		})}
	</div>

	<div class="flex items-center gap-1.5">
		<input type="checkbox" bind:checked={deleteFolderContents} />

		<div class="text-xs text-gray-500">
			{$i18n.t('Delete all contents inside this folder')}
		</div>
	</div>
</DeleteConfirmDialog>

<FolderModal bind:show={showFolderModal} edit={true} {folderId} onSubmit={updateHandler} />

{#if dragged && x && y}
	<DragGhost {x} {y}>
		<div class=" bg-black/80 backdrop-blur-2xl px-2 py-1 rounded-lg w-fit max-w-40">
			<div class="flex items-center gap-1">
				<FolderOpen className="size-3.5" strokeWidth="2" />
				<div class=" text-xs text-white line-clamp-1">
					{folders[folderId].name}
				</div>
			</div>
		</div>
	</DragGhost>
{/if}

<div bind:this={folderElement} class="relative {className}" draggable="true">
	{#if draggedOver}
		<div
			class="absolute top-0 left-0 w-full h-full rounded-xs bg-gray-100/50 dark:bg-gray-700/20 bg-opacity-50 dark:bg-opacity-10 z-50 pointer-events-none touch-none"
		></div>
	{/if}

	<Collapsible
		bind:open
		className="w-full"
		buttonClassName="w-full"
		onChange={(state) => {
			dispatch('open', state);
		}}
	>
		<!-- svelte-ignore a11y-no-static-element-interactions -->
		<div class="w-full group">
			<div
				id="folder-{folderId}-button"
				class="relative w-full py-1.5 px-2 rounded-md flex items-center gap-1.5 text-xm text-gray-900 dark:text-gray-100 font-medium hover:bg-gray-100 dark:hover:bg-gray-900 transition {$selectedFolder?.id === 
				folderId
					? 'bg-gray-100 dark:bg-gray-900 selected'
					: ''}"
				on:dblclick={(e) => {
					if (clickTimer) {
						clearTimeout(clickTimer); // cancel the single-click action
						clickTimer = null;
					}
					renameHandler();
				}}
				on:click={async (e) => {
					(e) => e.stopPropagation();
					if (clickTimer) {
						clearTimeout(clickTimer);
						clickTimer = null;
					}

					clickTimer = setTimeout(async () => {
						const folder = await getFolderById(localStorage.token, folderId).catch((error) => {
							toast.error(`${error}`);
							return null;
						});

						if (folder) {
							await selectedFolder.set(folder);
						}

						await goto('/');

						if ($mobile) {
							showSidebar.set(!$showSidebar);
						}
						clickTimer = null;
					}, 100); // 100ms delay (typical double-click threshold)
				}}
				on:pointerup={(e) => {
					e.stopPropagation();
				}}
			>
				<button
					class="text-gray-500 dark:text-gray-500 transition-all p-1 hover:bg-gray-200 dark:hover:bg-gray-850 rounded-lg"
					on:click={(e) => {
						e.stopPropagation();
						e.stopImmediatePropagation();
						open = !open;
						isExpandedUpdateDebounceHandler();
					}}
				>
					{#if folders[folderId]?.meta?.icon}
						<div class="flex group-hover:hidden transition-all">
							<Emoji className="size-3.5" shortCode={folders[folderId].meta.icon} />
						</div>

						<div class="hidden group-hover:flex transition-all p-[1px]">
							{#if open}
								<ChevronDown className=" size-3" strokeWidth="2.5" />
							{:else}
								<ChevronRight className=" size-3" strokeWidth="2.5" />
							{/if}
						</div>
					{:else}
						<div class="p-[1px]">
							{#if open}
								<ChevronDown className=" size-3" strokeWidth="2.5" />
							{:else}
								<ChevronRight className=" size-3" strokeWidth="2.5" />
							{/if}
						</div>
					{/if}
				</button>

				<div class="flex items-center justify-center group-disabled:opacity-50 group-data-disabled:opacity-50 icon"><button class="icon" data-state="closed"><div class="[&amp;_path]:stroke-current text-token-text-primary " style="width: 20px; height: 20px;"><div><svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 20 20" width="20" height="20" class="text-gray-900 dark:text-gray-100" preserveAspectRatio="xMidYMid meet" style="width: 100%; height: 100%; transform: translate3d(0px, 0px, 0px); content-visibility: visible; "><defs><clipPath id="__lottie_element_82"><rect width="20" height="20" x="0" y="0"></rect></clipPath></defs><g clip-path="url(#__lottie_element_82)"><g transform="matrix(1,0,0,1,0,0)" opacity="1" style="display: block;"><g opacity="1" transform="matrix(1,0,0,1,2.75,3.5)"><path stroke-linecap="round" stroke-linejoin="miter" fill-opacity="0" stroke-miterlimit="4" stroke="currentColor" stroke-opacity="1" stroke-width="1.33" d=" M14.5,6 C14.5,5.808333396911621 14.5,5.616666793823242 14.5,5.425000190734863 C14.5,4.094880104064941 14.5,3.429810047149658 14.241100311279297,2.9217700958251953 C14.013400077819824,2.4748899936676025 13.650099754333496,2.111560106277466 13.203200340270996,1.8838599920272827 C12.695199966430664,1.625 12.030099868774414,1.625 10.699999809265137,1.625 C9.931366920471191,1.625 9.16273307800293,1.625 8.394100189208984,1.625 C8.242400169372559,1.625 8.166600227355957,1.625 8.093500137329102,1.6204999685287476 C7.561999797821045,1.5877100229263306 7.056839942932129,1.377210021018982 6.6593098640441895,1.0228099822998047 C6.604680061340332,0.974120020866394 6.551270008087158,0.9202499985694885 6.444439888000488,0.8125 C6.444439888000488,0.8125 6.444439888000488,0.8125 6.444439888000488,0.8125 C6.337619781494141,0.7047500014305115 6.284210205078125,0.650879979133606 6.229579925537109,0.6021900177001953 C5.83204984664917,0.24778999388217926 5.326930046081543,0.037289999425411224 4.795370101928711,0.0044999998062849045 C4.722330093383789,0 4.646470069885254,0 4.494740009307861,0 C4.263160228729248,0 4.031579971313477,0 3.799999952316284,0 C2.4698801040649414,0 1.8048100471496582,0 1.2967699766159058,0.2588599920272827 C0.8498899936676025,0.48655998706817627 0.48655998706817627,0.8498899936676025 0.2588599920272827,1.2967699766159058 C0,1.8048100471496582 0,2.469870090484619 0,3.799999952316284 C0,5.599999904632568 0,7.400000095367432 0,9.199999809265137 C0,10.530099868774414 0,11.195199966430664 0.2588599920272827,11.703200340270996 C0.48655998706817627,12.150099754333496 0.8498899936676025,12.513400077819824 1.2967699766159058,12.741100311279297 C1.8048100471496582,13 2.4698801040649414,13 3.799999952316284,13 C4.616666793823242,13 5.433333396911621,13 6.25,13"></path></g></g><g transform="matrix(1,0,0,1,0,0)" opacity="1" style="display: block;"><g opacity="1" transform="matrix(1,0,0,1,2.75,8.913877487182617)"><path stroke-linecap="round" stroke-linejoin="miter" fill-opacity="0" stroke-miterlimit="4" stroke="currentColor" stroke-opacity="1" stroke-width="1.33" d=" M0.5920000076293945,0 C0.38477998971939087,0 0.2811700105667114,0 0.20202000439167023,0.03922419250011444 C0.1324000060558319,0.07372163981199265 0.07580000162124634,0.12877944111824036 0.04033000022172928,0.19649054110050201 C0,0.27346059679985046 0,0.37422969937324524 0,0.5757679343223572 C0,1.6806167364120483 0,2.7854654788970947 0,3.8903141021728516 C0,5.183944225311279 0,5.830807685852051 0.2588599920272827,6.324878692626953 C0.48655998706817627,6.759525299072266 0.8498899936676025,7.112961292266846 1.2967699766159058,7.334417819976807 C1.8048100471496582,7.586122035980225 2.4698801040649414,7.586122035980225 3.799999952316284,7.586122035980225 C6.099999904632568,7.586122035980225 8.399999618530273,7.586122035980225 10.699999809265137,7.586122035980225 C12.030099868774414,7.586122035980225 12.695199966430664,7.586122035980225 13.203200340270996,7.334417819976807 C13.650099754333496,7.112961292266846 14.013400077819824,6.759525299072266 14.241100311279297,6.324878692626953 C14.5,5.830807685852051 14.5,5.183944225311279 14.5,3.8903141021728516 C14.5,2.7854654788970947 14.5,1.6806167364120483 14.5,0.5757679343223572 C14.5,0.37422969937324524 14.5,0.27346059679985046 14.459699630737305,0.19649054110050201 C14.424200057983398,0.12877944111824036 14.367600440979004,0.07372163981199265 14.29800033569336,0.03922419250011444 C14.218799591064453,0 14.11520004272461,0 13.907999992370605,0 C9.46933364868164,0 5.030666828155518,0 0.5920000076293945,0 C0.5920000076293945,0 0.5920000076293945,0 0.5920000076293945,0 C0.5920000076293945,0 0.5920000076293945,0 0.5920000076293945,0z"></path></g></g></g></svg></div></div></button></div>

				<div class="translate-y-[0.5px] flex-1 justify-start text-start line-clamp-1">
					{#if edit}
						<input
							id="folder-{folderId}-input"
							type="text"
							bind:value={name}
							on:blur={() => {
								console.log('Blur');
								updateHandler({ name });
								edit = false;
							}}
							on:click={(e) => {
								// Prevent accidental collapse toggling when clicking inside input
								e.stopPropagation();
							}}
							on:mousedown={(e) => {
								// Prevent accidental collapse toggling when clicking inside input
								e.stopPropagation();
							}}
							on:keydown={(e) => {
								if (e.key === 'Enter') {
									updateHandler({ name });
									edit = false;
								}
							}}
							class="w-full h-full bg-transparent outline-hidden"
						/>
					{:else}
						{folders[folderId].name}
					{/if}
				</div>

				<button
					class="absolute z-10 right-2 invisible group-hover:visible self-center flex items-center dark:text-gray-300"
				>
					<FolderMenu
						onEdit={() => {
							showFolderModal = true;
						}}
						onDelete={() => {
							showDeleteConfirm = true;
						}}
						onExport={() => {
							exportHandler();
						}}
					>
						<div class="p-1 dark:hover:bg-gray-850 rounded-lg touch-auto">
							<EllipsisHorizontal className="size-4" strokeWidth="2.5" />
						</div>
					</FolderMenu>
				</button>
			</div>
		</div>

		<div slot="content" class="w-full">
			{#if (folders[folderId]?.childrenIds ?? []).length > 0 || (chats ?? []).length > 0}
				<div
					class="ml-3 pl-1 mt-[1px] flex flex-col overflow-y-auto scrollbar-hidden border-s border-gray-100 dark:border-gray-900"
				>
					{#if folders[folderId]?.childrenIds}
						{@const children = folders[folderId]?.childrenIds
							.map((id) => folders[id])
							.sort((a, b) =>
								a.name.localeCompare(b.name, undefined, {
									numeric: true,
									sensitivity: 'base'
								})
							)}

						{#each children as childFolder (`${folderId}-${childFolder.id}`)}
							<svelte:self
								bind:folderRegistry
								{folders}
								folderId={childFolder.id}
								{shiftKey}
								parentDragged={dragged}
								{onItemMove}
								{onDelete}
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
					{/if}

					{#each chats ?? [] as chat (chat.id)}
						<ChatItem
							id={chat.id}
							title={chat.title}
							{shiftKey}
							on:change={(e) => {
								dispatch('change', e.detail);
							}}
						/>
					{/each}
				</div>
			{/if}

			{#if chats === null}
				<div class="flex justify-center items-center p-2">
					<Spinner className="size-4 text-gray-500" />
				</div>
			{/if}
		</div>
	</Collapsible>
</div>
