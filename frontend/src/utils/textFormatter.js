// src/utils/textFormatter.js
export function formatMarkdown(text) {
  return text
    .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>") // **볼드**
    .replace(/\*(.*?)\*/g, "<em>$1</em>") // *이탤릭*
    .replace(/\n/g, "<br/>"); // 줄바꿈
}
