document.addEventListener("DOMContentLoaded", () => {
    const container = document.getElementById("demo-section");
    if (!container) return;
  
    fetch("samples.json")
      .then((res) => {
        if (!res.ok) {
          throw new Error(`HTTP error! status: ${res.status}`);
        }
        return res.json();
      })
      .then((samples) => {
        if (!Array.isArray(samples)) {
          container.innerHTML = "<p>Error: samples.json is not an array.</p>";
          return;
        }
  
        let html = "";
        samples.forEach((item, i) => {
          const index = i + 1;
  
          // Extract fields
          const refAudio        = item.reference_audio      || "";
          const deepAudio       = item.mars6_deep_audio     || "";
          const shallowAudio    = item.mars6_shallow_audio  || "";
          const metavoiceAudio  = item.metavoice_1b_audio   || "";
          const styletts2Audio  = item.styletts2_audio      || "";
          const xttsv2Audio     = item.xttsv2_audio         || "";
          const refTranscript   = item.reference_transcript || "";
          const tgtTranscript   = item.target_transcript    || "";
  
          // We'll create a card for each sample:
          html += `
            <div class="card mb-4 shadow-sm">
              <div class="card-body">
                <h5 class="card-title mb-3">Sample #${index}</h5>
                <div class="table-responsive">
                  <table class="table table-sm align-middle" style="margin-bottom:1rem;">
                    <thead>
                      <tr class="table-light">
                        <th scope="col" style="width:16%;">Reference</th>
                        <th scope="col" style="width:16%;">MARS6-Deep-Clone</th>
                        <th scope="col" style="width:16%;">MARS6-Shallow-Clone</th>
                        <th scope="col" style="width:16%;">Metavoice-1B</th>
                        <th scope="col" style="width:16%;">StyleTTS2</th>
                        <th scope="col" style="width:16%;">XTTSv2</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr>
                        <td>
                          <audio controls preload="none">
                            <source src="${refAudio}" type="audio/flac">
                            Your browser does not support the audio element.
                          </audio>
                        </td>
                        <td>
                          <audio controls preload="none">
                            <source src="${deepAudio}" type="audio/flac">
                            Your browser does not support the audio element.
                          </audio>
                        </td>
                        <td>
                          <audio controls preload="none">
                            <source src="${shallowAudio}" type="audio/flac">
                            Your browser does not support the audio element.
                          </audio>
                        </td>
                        <td>
                          <audio controls preload="none">
                            <source src="${metavoiceAudio}" type="audio/flac">
                            Your browser does not support the audio element.
                          </audio>
                        </td>
                        <td>
                          <audio controls preload="none">
                            <source src="${styletts2Audio}" type="audio/flac">
                            Your browser does not support the audio element.
                          </audio>
                        </td>
                        <td>
                          <audio controls preload="none">
                            <source src="${xttsv2Audio}" type="audio/flac">
                            Your browser does not support the audio element.
                          </audio>
                        </td>
                      </tr>
                    </tbody>
                  </table>
                </div>
                <div class="mb-2">
                  <strong>Reference Transcript:</strong> 
                  <span class="text-body">${refTranscript}</span>
                </div>
                <div>
                  <strong>Target/Generation Transcript:</strong> 
                  <span class="text-body">${tgtTranscript}</span>
                </div>
              </div>
            </div>
          `;
        });
  
        container.innerHTML = html;
      })
      .catch((err) => {
        console.error("Error loading samples.json:", err);
        container.innerHTML = `<p style="color:red;">Could not load sample data. Check console for details.</p>`;
      });
  });  