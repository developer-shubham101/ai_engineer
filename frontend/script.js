async function sendMessage() {
  const input = document.getElementById("user-input");
  const chatWindow = document.getElementById("chat-window");

  const text = input.value.trim();
  if (!text) return;

  // Add user message to chat window
  chatWindow.innerHTML += `<div class='message user'><strong>You:</strong> ${text}</div>`;
  chatWindow.scrollTop = chatWindow.scrollHeight;

  input.value = "";

  // Call FastAPI LLM endpoint
  const response = await fetch("http://localhost:8000/api/local/query", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      question: text,
      top_k: 3,
      use_llm: true
    })
  });

  const data = await response.json();
  const answer = data.answer || "(no response)";

  // Add bot response to chat window
  chatWindow.innerHTML += `<div class='message bot'><strong>Bot:</strong> ${answer}</div>`;
  chatWindow.scrollTop = chatWindow.scrollHeight;
}
