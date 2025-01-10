document.addEventListener("DOMContentLoaded", () => {
  const container = document.getElementById("demo-section");
  if (!container) return;

  fetch("samples.json")
    .then((res) => {
      if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
      return res.json();
    })
    .then((samples) => {
      if (!Array.isArray(samples)) {
        container.innerHTML = "<p>Error: samples.json is not an array.</p>";
        return;
      }

      // If narrow screen (<768px), show horizontal scrolling. Otherwise, single big table.
      if (window.innerWidth < 768) {
        container.innerHTML = buildScrollingLayout(samples);
      } else {
        container.innerHTML = buildTableLayout(samples);
      }
    })
    .catch((err) => {
      console.error("Error loading samples.json:", err);
      container.innerHTML = `<p style="color:red;">Could not load sample data. Check console for details.</p>`;
    });
});

/**
 * For wide screens: a single large table with column headings, 
 * each sample has a row of audio players + a second row for transcripts.
 */
function buildTableLayout(samples) {
  let html = `
    <div class="table-responsive mb-4">
      <table class="table table-bordered table-sm align-middle text-center">
        <thead class="table-light">
          <tr>
            <th style="width:16%;">Reference</th>
            <th style="width:16%;">MARS6-Deep</th>
            <th style="width:16%;">MARS6-Shallow</th>
            <th style="width:16%;">Metavoice-1B</th>
            <th style="width:16%;">StyleTTS2</th>
            <th style="width:16%;">XTTSv2</th>
          </tr>
        </thead>
        <tbody>
  `;

  samples.forEach(item => {
    const refAudio       = item.reference_audio       || "";
    const deepAudio      = item.mars6_deep_audio      || "";
    const shallowAudio   = item.mars6_shallow_audio   || "";
    const metaVoiceAudio = item.metavoice_1b_audio    || "";
    const styletts2Audio = item.styletts2_audio       || "";
    const xttsv2Audio    = item.xttsv2_audio          || "";
    const refTranscript  = item.reference_transcript  || "(No reference transcript)";
    const tgtTranscript  = item.target_transcript     || "(No target transcript)";

    // 1 row for the audios
    html += `
      <tr>
        <td><audio controls preload="none" style="width:100%;"><source src="${refAudio}" type="audio/flac"></audio></td>
        <td><audio controls preload="none" style="width:100%;"><source src="${deepAudio}" type="audio/flac"></audio></td>
        <td><audio controls preload="none" style="width:100%;"><source src="${shallowAudio}" type="audio/flac"></audio></td>
        <td><audio controls preload="none" style="width:100%;"><source src="${metaVoiceAudio}" type="audio/flac"></audio></td>
        <td><audio controls preload="none" style="width:100%;"><source src="${styletts2Audio}" type="audio/flac"></audio></td>
        <td><audio controls preload="none" style="width:100%;"><source src="${xttsv2Audio}" type="audio/flac"></audio></td>
      </tr>
      <tr>
        <td colspan="6">
          <p class="text-muted transcript-text mb-0">
            <strong>Reference:</strong> ${refTranscript}<br>
            <strong>Target:</strong> ${tgtTranscript}
          </p>
        </td>
      </tr>
    `;
  });

  // close table
  html += `
        </tbody>
      </table>
    </div>
  `;
  return html;
}

/**
 * For narrow screens: horizontally scrollable layout. 
 * STILL includes headings above each audio player.
 */
function buildScrollingLayout(samples) {
  // We'll create a single container for ALL samples or 
  // we can do one container per sample. 
  // Typically, we do one container per sample so transcripts can be separate.
  let html = `<div class="mb-4">`;

  samples.forEach((item) => {
    const refAudio       = item.reference_audio       || "";
    const deepAudio      = item.mars6_deep_audio      || "";
    const shallowAudio   = item.mars6_shallow_audio   || "";
    const metaVoiceAudio = item.metavoice_1b_audio    || "";
    const styletts2Audio = item.styletts2_audio       || "";
    const xttsv2Audio    = item.xttsv2_audio          || "";
    const refTranscript  = item.reference_transcript  || "(No reference transcript)";
    const tgtTranscript  = item.target_transcript     || "(No target transcript)";

    html += `
      <div class="card mb-3 shadow-sm">
        <div class="card-body" style="overflow-x:auto; white-space:nowrap;">
          <!-- 6 columns horizontally, each with a heading + audio -->
          <span style="display:inline-block; width:180px; margin-right:1rem; vertical-align:top;">
            <p class="fw-bold mb-1 text-center">Reference</p>
            <audio controls preload="none" style="width:100%;">
              <source src="${refAudio}" type="audio/flac">
            </audio>
          </span>
          <span style="display:inline-block; width:180px; margin-right:1rem; vertical-align:top;">
            <p class="fw-bold mb-1 text-center">MARS6-Deep</p>
            <audio controls preload="none" style="width:100%;">
              <source src="${deepAudio}" type="audio/flac">
            </audio>
          </span>
          <span style="display:inline-block; width:180px; margin-right:1rem; vertical-align:top;">
            <p class="fw-bold mb-1 text-center">MARS6-Shallow</p>
            <audio controls preload="none" style="width:100%;">
              <source src="${shallowAudio}" type="audio/flac">
            </audio>
          </span>
          <span style="display:inline-block; width:180px; margin-right:1rem; vertical-align:top;">
            <p class="fw-bold mb-1 text-center">Metavoice-1B</p>
            <audio controls preload="none" style="width:100%;">
              <source src="${metaVoiceAudio}" type="audio/flac">
            </audio>
          </span>
          <span style="display:inline-block; width:180px; margin-right:1rem; vertical-align:top;">
            <p class="fw-bold mb-1 text-center">StyleTTS2</p>
            <audio controls preload="none" style="width:100%;">
              <source src="${styletts2Audio}" type="audio/flac">
            </audio>
          </span>
          <span style="display:inline-block; width:180px; margin-right:1rem; vertical-align:top;">
            <p class="fw-bold mb-1 text-center">XTTSv2</p>
            <audio controls preload="none" style="width:100%;">
              <source src="${xttsv2Audio}" type="audio/flac">
            </audio>
          </span>
        </div>
        <div>
          <p class="text-muted transcript-text mb-0">
            <strong>Reference:</strong> ${refTranscript}<br>
            <strong>Target:</strong> ${tgtTranscript}
          </p>
        </div>
      </div>
    `;
  });

  html += `</div>`;
  return html;
}