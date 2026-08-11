/**
 * FRONTEND LOGIC
 * ================
 * Connects to the /stream/query SSE endpoint and updates the page live as
 * each node in the LangGraph agent finishes -- instead of one long silent
 * wait, the user sees "Classifying your question..." -> "Searching FOMC
 * documents..." -> etc. as they happen.
 *
 * WHY FETCH + ReadableStream INSTEAD OF THE EventSource API?
 * The browser's built-in EventSource only supports GET requests. Our
 * endpoint needs to receive the question in a POST body, so we manually
 * read the streamed response body instead -- this is a common workaround
 * when you need SSE-style streaming with a POST request.
 */

const API_BASE = "http://localhost:8000";

const form = document.getElementById("query-form");
const input = document.getElementById("question-input");
const progressList = document.getElementById("progress-list");
const answerBox = document.getElementById("answer-box");
const answerText = document.getElementById("answer-text");
const groundedBadge = document.getElementById("grounded-badge");
const confidenceBadge = document.getElementById("confidence-badge");
const retriesBadge = document.getElementById("retries-badge");
const sourcesBox = document.getElementById("sources-box");

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const question = input.value.trim();
  if (!question) return;

  progressList.innerHTML = "";
  answerBox.classList.add("hidden");
  form.querySelector("button").disabled = true;

  try {
    const response = await fetch(`${API_BASE}/stream/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      // SSE messages are separated by a blank line ("\n\n"). Split on
      // that to process each complete event as it arrives.
      const parts = buffer.split("\n\n");
      buffer = parts.pop(); // keep the last (possibly incomplete) chunk

      for (const part of parts) {
        if (!part.startsWith("data: ")) continue;
        const payload = JSON.parse(part.slice(6));
        handleEvent(payload);
      }
    }
  } catch (err) {
    progressList.innerHTML = `<div class="progress-item">Error: ${err.message}. Is the API running at ${API_BASE}?</div>`;
  } finally {
    form.querySelector("button").disabled = false;
  }
});

function handleEvent(payload) {
  if (payload.type === "progress") {
    const item = document.createElement("div");
    item.className = "progress-item done";
    item.textContent = payload.message;
    progressList.appendChild(item);
  } else if (payload.type === "final") {
    renderFinalAnswer(payload);
  }
}

function renderFinalAnswer(payload) {
  answerBox.classList.remove("hidden");
  answerText.textContent = payload.answer;

  groundedBadge.textContent = payload.is_grounded ? "Grounded" : "Not fully grounded";
  groundedBadge.className = payload.is_grounded ? "badge-grounded" : "badge-not-grounded";

  confidenceBadge.textContent = `Confidence: ${Math.round(payload.confidence * 100)}%`;
  retriesBadge.textContent = `Retries used: ${payload.retries_used}`;

  sourcesBox.textContent = payload.sources.length
    ? `Sources: ${payload.sources.join(", ")}`
    : "No sources cited";
}
