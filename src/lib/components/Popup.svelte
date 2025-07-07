<script>
  import { createEventDispatcher } from "svelte";

  const dispatch = createEventDispatcher();

  function closeModal() {
    dispatch("close");
  }

  function closeForDay() {
    dispatch("close-for-day");
  }
</script>

<div
  class="fixed inset-0 z-50 flex items-center justify-center"
  on:click={closeModal}
>
  <div
    class="transform rounded-2xl border border-gray-300 bg-white p-6 text-left shadow-xl transition-all mx-4 sm:my-8 sm:w-full sm:max-w-lg"
    on:click|stopPropagation
  >
    <h3 class="text-lg font-medium leading-6 text-gray-900">
      사이트 사용 관련 안내
    </h3>
    <div class="mt-2 space-y-2">
      <p class="text-sm text-gray-800">
        <strong>1. KCD 자료와 암등록 자료 기반으로 관련 질문을 하시면 AI가 답변합니다</strong>
        <br />
        -좌측 상단 AI 종류를 확인하세요.
      </p>
      <p class="text-sm text-gray-800">
        <strong>2. AI가 잘못된 정보를 알려주면 해당 답변에 직접 평가해주세요</strong> 
        <br />
        -추후 AI 답변 개선에 활용하겠습니다. 
        <br />
        -평가방법은 좌측 사이드바 채널 #공지사항을 참고해주세요.
      </p>
      <p class="text-sm text-gray-800">
        <strong>3. ICD-9-CM(수술코드)은 아직 지원하지 않습니다.(추후 지원예정)</strong>
      </p>
      <p class="text-sm text-gray-800">
        <strong>4. 현재 웹사이트 테스트 중이므로 오류가 발생할 수 있습니다. </strong>
      </p>
    </div>
    <div class="mt-4 flex space-x-2">
      <button
        type="button"
        class="inline-flex justify-center rounded-md border border-transparent bg-blue-100 px-4 py-2 text-sm font-medium text-blue-900 hover:bg-blue-200 focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 focus-visible:ring-offset-2"
        on:click={closeModal}
      >
        닫기
      </button>
      <button
        type="button"
        class="inline-flex justify-center rounded-md border border-gray-300 bg-white px-4 py-2 text-sm font-medium text-gray-700 hover:bg-gray-50 focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 focus-visible:ring-offset-2"
        on:click={closeForDay}
      >
        하루 동안 보지 않기
      </button>
    </div>
  </div>
</div>
