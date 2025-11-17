// Paste into js/app.js

/*
  app.js
  Frontend logic (separated from HTML/CSS)
  IMPORTANT: Change BASE_API_URL below if your backend runs elsewhere.
*/

const BASE_API_URL = "http://192.168.1.2:8000/api/local"; // <-- change if needed

$(function () {
  // UI state
  const state = {
    apiKey: "",
    role: "Employee",
    useLlm: false,
    topK: 3,
    history: [], // {role:'user'|'bot', text, answer, ...}
    inFlight: false,
    selectedFile: null,
    accessRequestsLocal: []
  };

  // Load saved API key if remember checked
  if (localStorage.getItem('remember_api_key') === 'true') {
    const saved = localStorage.getItem('api_key');
    if (saved) {
      $('#apiKeyInput').val(saved);
      $('#rememberKeyCheckbox').prop('checked', true);
      $('#rememberKeyBtn').addClass('btn-success');
      state.apiKey = saved;
    }
  }

  // Load history from localStorage
  try {
    const hs = localStorage.getItem('chat_history_v1');
    if (hs) {
      state.history = JSON.parse(hs);
      state.history.forEach(h => renderMessage(h, true));
      updateSessionUI();
    }
  } catch (e) { console.warn('history load failed', e); }

  // Initialize controls
  $('#roleSelect').val(state.role);
  $('#displayTopK').text(state.topK);
  $('#displayUseLlm').text('off');

  // Utility: show toast
  function showToast(message, type = 'danger', title = '') {
    const id = 't' + Date.now();
    const toastHtml = `
      <div id="${id}" class="toast align-items-center text-bg-${type} border-0" role="alert" aria-live="assertive" aria-atomic="true">
        <div class="d-flex">
          <div class="toast-body">
            ${title ? ('<strong>' + title + ':</strong> ') : ''}${message}
          </div>
          <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast" aria-label="Close"></button>
        </div>
      </div>`;
    $('#toastContainer').append(toastHtml);
    const t = new bootstrap.Toast(document.getElementById(id), { delay: 5000 });
    t.show();
    document.getElementById(id).addEventListener('hidden.bs.toast', () => {
      $('#' + id).remove();
    });
  }

  // update session UI counts
  function updateSessionUI() {
    $('#msgCount').text(state.history.length);
    $('#lastActivity').text(state.history.length ? state.history[state.history.length - 1].ts : '-');
  }

  // Save history to localStorage
  function persistHistory() {
    localStorage.setItem('chat_history_v1', JSON.stringify(state.history));
  }

  // Save API key if remember is checked
  function maybeSaveApiKey() {
    if ($('#rememberKeyCheckbox').is(':checked')) {
      localStorage.setItem('api_key', state.apiKey || '');
      localStorage.setItem('remember_api_key', 'true');
      $('#rememberKeyBtn').addClass('btn-success');
    } else {
      localStorage.removeItem('api_key');
      localStorage.setItem('remember_api_key', 'false');
      $('#rememberKeyBtn').removeClass('btn-success');
    }
  }

  // Render a message in the chat window.
  function renderMessage(msg, skipAnim = false) {
    const $messages = $('#messages');
    const ts = msg.ts || new Date().toLocaleString();
    const isUser = msg.role === 'user';
    const $wrap = $('<div>').addClass('message ' + (isUser ? 'msg-user justify-content-end' : 'msg-bot'));
    const $bubble = $('<div>').addClass('bubble ' + (isUser ? 'bubble-user' : 'bubble-bot'));
    const $top = $('<div>').append($('<div>').addClass('small-muted mb-1').text((isUser ? 'You' : 'Assistant') + ' • ' + ts));
    $bubble.append($top);

    const mainText = isUser ? (msg.text || '') : (msg.answer || msg.text || '');
    $bubble.append($('<div>').addClass('mb-2').html(escapeHtml(mainText).replace(/\n/g, '<br>')));
    if (!isUser && msg.context) {
      $bubble.append($('<div>').addClass('meta-small mt-1').text('Context: ' + truncate(msg.context, 300)));
    }

    if (!isUser) {
      const $extras = $('<div>').addClass('mt-2');

      if (Array.isArray(msg.retrieved) && msg.retrieved.length > 0) {
        const $retrBtn = $('<div>').append(
          $('<button>').addClass('btn btn-sm btn-outline-secondary me-2').text('Show Retrieved (' + msg.retrieved.length + ')')
            .on('click', function () {
              $(this).toggleClass('active');
              $retrList.toggle();
            })
        );
        $extras.append($retrBtn);

        const $retrList = $('<div>').addClass('mt-2').hide();
        msg.retrieved.forEach(d => {
          const meta = d.metadata || {};
          const $s = $('<div>').addClass('retrieved-snippet');
          $s.append($('<div>').addClass('meta-small').text('Source: ' + (d.source || meta.source_name || meta.department || 'unknown')));
          $s.append($('<div>').html(escapeHtml(truncate(d.text || d.preview || '', 600)).replace(/\n/g, '<br>')));
          $s.append($('<div>').addClass('meta-small mt-1').text('Metadata: ' + JSON.stringify(meta)));
          if (meta && (meta.sensitivity === 'restricted' || meta.sensitivity === 'confidential')) {
            const $reqBtn = $('<button>').addClass('btn btn-sm btn-outline-primary mt-2').text('Request Access').on('click', function () {
              openRequestAccessModal(d.id || d.document_id || d.chunk_id, d.source || meta.source_name);
            });
            $s.append($reqBtn);
          }
          $retrList.append($s);
        });
        $extras.append($retrList);
      }

      if (msg.filtered_out_count && msg.filtered_out_count > 0) {
        const badgeText = msg.filtered_out_count + ' results were filtered — show public summaries';
        const $badge = $('<span>').addClass('badge bg-warning text-dark badge-filtered ms-2').text(badgeText)
          .on('click', function () {
            showPublicSummaries(msg.public_summaries || []);
          });
        $extras.append($badge);
      }

      const role = $('#roleSelect').val();
      if (!isUser && msg.filtered_details && (role === 'Executive' || role === 'Legal')) {
        const $fd = $('<details class="mt-2"><summary>Filtered details (admin)</summary><pre>' + escapeHtml(JSON.stringify(msg.filtered_details, null, 2)) + '</pre></details>');
        $extras.append($fd);
      }

      if ($extras.children().length) $bubble.append($extras);
    }

    $wrap.append($bubble);
    if (skipAnim) {
      $messages.append($wrap);
    } else {
      $wrap.hide();
      $messages.append($wrap);
      $wrap.slideDown(250).delay(50).animate({ opacity: 1 }, { duration: 120 });
    }
    $messages.stop().animate({ scrollTop: $messages[0].scrollHeight }, 300);
  }

  // Utility: truncate & escape
  function truncate(text, max = 200) {
    if (!text) return '';
    text = String(text);
    if (text.length <= max) return text;
    return text.slice(0, max) + '...';
  }
  function escapeHtml(str) {
    if (str === undefined || str === null) return '';
    return String(str).replace(/[&<>"']/g, function (m) { return ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[m]; });
  }

  function showPublicSummaries(summaries) {
    const text = (summaries && summaries.length) ? summaries.map((s, i) => '<div class="mb-2"><strong>Summary ' + (i + 1) + ':</strong><div>' + escapeHtml(s).replace(/\n/g, '<br>') + '</div></div>').join('') : '<div class="small-muted">No public summaries available.</div>';
    const modalHtml = '<div class="modal fade" id="publicSummariesModal" tabindex="-1" aria-hidden="true"><div class="modal-dialog"><div class="modal-content"><div class="modal-header"><h5 class="modal-title">Public Summaries</h5><button type="button" class="btn-close" data-bs-dismiss="modal"></button></div><div class="modal-body">' + text + '</div><div class="modal-footer"><button class="btn btn-secondary" data-bs-dismiss="modal">Close</button></div></div></div></div>';
    $('body').append(modalHtml);
    const m = new bootstrap.Modal(document.getElementById('publicSummariesModal'));
    m.show();
    document.getElementById('publicSummariesModal').addEventListener('hidden.bs.modal', function () { $('#publicSummariesModal').remove(); });
  }

  function openRequestAccessModal(docId, sourceName) {
    $('#request_doc_id').val(docId || '');
    $('#request_reason').val('Please grant access to review the document: ' + (sourceName || docId));
    $('#requestAccessResponse').empty();
    const m = new bootstrap.Modal(document.getElementById('requestAccessModal'));
    m.show();
  }

  // Send message handlers
  $('#sendBtn').on('click', submitMessage);
  $('#composerInput').on('keydown', function (e) {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
      submitMessage();
      e.preventDefault();
    }
  });

  function disableSend(flag) {
    $('#sendBtn').prop('disabled', flag);
    state.inFlight = flag;
  }

  function submitMessage() {
    if (state.inFlight) return;
    const question = $('#composerInput').val().trim();
    if (!question) return showToast('Please enter a question', 'warning', 'Validation');
    state.apiKey = $('#apiKeyInput').val().trim();
    state.role = $('#roleSelect').val();
    state.useLlm = $('#useLlmSwitch').is(':checked');
    state.topK = parseInt($('#topKInput').val()) || 3;

    if (!state.apiKey) return showToast('API key is required', 'warning', 'Auth');

    disableSend(true);

    const userMsg = { role: 'user', text: question, ts: new Date().toLocaleString() };
    state.history.push(userMsg);
    renderMessage(userMsg);
    persistHistory();
    updateSessionUI();

    const payload = {
      question: question,
      top_k: state.topK,
      use_llm: state.useLlm
    };

    const url = BASE_API_URL + '/query';
    const headers = { 'Content-Type': 'application/json', 'X-API-Key': state.apiKey };

    fetch(url, { method: 'POST', headers, body: JSON.stringify(payload) })
      .then(async res => {
        if (!res.ok) {
          const txt = await res.text().catch(() => res.statusText);
          throw new Error('Server error: ' + res.status + ' - ' + txt);
        }
        return res.json();
      })
      .then(data => {
        const botMsg = {
          role: 'bot',
          answer: data.answer || 'No answer returned',
          retrieved: data.retrieved || [],
          context: data.context || '',
          filtered_out_count: data.filtered_out_count || 0,
          public_summaries: data.public_summaries || [],
          filtered_details: data.filtered_details || null,
          ts: new Date().toLocaleString()
        };
        state.history.push(botMsg);
        renderMessage(botMsg);
        persistHistory();
        updateSessionUI();

        if (botMsg.filtered_out_count && botMsg.filtered_out_count > 0) {
          showToast(botMsg.filtered_out_count + ' results were filtered for your role. See public summaries or request access.', 'warning', 'Filtered');
        }
      })
      .catch(err => {
        showToast(err.message || 'Network error', 'danger', 'Error');
        const errMsg = { role: 'bot', answer: 'Error: ' + (err.message || 'Network error'), ts: new Date().toLocaleString() };
        state.history.push(errMsg);
        renderMessage(errMsg);
        persistHistory();
        updateSessionUI();
      })
      .finally(() => {
        setTimeout(() => {
          disableSend(false);
        }, 2000);
      });

    $('#composerInput').val('');
  }

  // Request access submit
  $('#requestAccessForm').on('submit', function (e) {
    e.preventDefault();
    const docId = $('#request_doc_id').val() || undefined;
    const reason = $('#request_reason').val().trim();
    const payload = {};
    if (docId) payload.document_id = docId;
    payload.reason = reason;
    payload.source_name = undefined;

    const headers = { 'Content-Type': 'application/json', 'X-API-Key': $('#apiKeyInput').val().trim() || '' };
    fetch(BASE_API_URL + '/request-access', { method: 'POST', headers, body: JSON.stringify(payload) })
      .then(async res => {
        if (!res.ok) {
          const txt = await res.text().catch(() => res.statusText);
          throw new Error('Server error: ' + res.status + ' - ' + txt);
        }
        return res.json();
      })
      .then(data => {
        $('#requestAccessResponse').html('<div class="alert alert-success small-muted">Request submitted.</div>');
        state.accessRequestsLocal.push({ id: 'local-' + Date.now(), document_id: docId, reason, status: 'pending', ts: new Date().toLocaleString() });
        renderLocalAccessRequests();
      })
      .catch(err => {
        $('#requestAccessResponse').html('<div class="alert alert-danger small-muted">Error: ' + escapeHtml(err.message) + '</div>');
      });
  });

  // Admin fetch access requests
  $('#btnFetchAccessRequests').on('click', function () {
    const key = $('#apiKeyInput').val().trim();
    if (!key) return showToast('API key required to fetch access requests', 'warning', 'Auth');
    $.ajax({
      url: BASE_API_URL + '/access-requests',
      method: 'GET',
      headers: { 'X-API-Key': key },
      success: function (data) { renderAccessRequests(data); },
      error: function (xhr) { showToast('Failed to fetch access requests: ' + (xhr.responseText || xhr.statusText), 'danger', 'Error'); }
    });
  });

  $('#btnShowAccessRequestsLocal').on('click', function () { renderLocalAccessRequests(); });

  function renderAccessRequests(list) {
    const $list = $('#accessRequestsList').empty();
    if (!Array.isArray(list) || !list.length) {
      $list.append($('<div>').addClass('small-muted').text('No requests found.'));
      return;
    }
    list.forEach(req => {
      const $item = $('<div>').addClass('list-group-item');
      $item.append($('<div>').addClass('d-flex w-100 justify-content-between').append($('<h6>').text(req.id || req.request_id || 'req')));
      $item.append($('<p>').addClass('mb-1').text(req.reason || JSON.stringify(req)));
      const $actions = $('<div>').addClass('mt-2');
      $actions.append($('<button>').addClass('btn btn-sm btn-success me-2').text('Approve').on('click', function () { handleApprove(req); }));
      $actions.append($('<button>').addClass('btn btn-sm btn-danger').text('Deny').on('click', function () { handleDeny(req); }));
      $item.append($actions);
      $list.append($item);
    });
  }

  function renderLocalAccessRequests() {
    const $list = $('#accessRequestsList').empty();
    if (!state.accessRequestsLocal.length) {
      $list.append($('<div>').addClass('small-muted').text('No local requests.'));
      return;
    }
    state.accessRequestsLocal.forEach(r => {
      const $item = $('<div>').addClass('list-group-item');
      $item.append($('<div>').addClass('d-flex w-100 justify-content-between').append($('<h6>').text(r.document_id || r.id)));
      $item.append($('<p>').addClass('mb-1').text(r.reason));
      $item.append($('<small>').addClass('text-muted').text('Status: ' + r.status + ' • ' + r.ts));
      const $actions = $('<div>').addClass('mt-2');
      $actions.append($('<button>').addClass('btn btn-sm btn-success me-2').text('Approve (local)').on('click', function () { r.status = 'approved'; renderLocalAccessRequests(); }));
      $actions.append($('<button>').addClass('btn btn-sm btn-danger').text('Deny (local)').on('click', function () { r.status = 'denied'; renderLocalAccessRequests(); }));
      $item.append($actions);
      $list.append($item);
    });
  }

  function handleApprove(req) { showToast('Approved request ' + (req.id || ''), 'success', 'Admin'); }
  function handleDeny(req) { showToast('Denied request ' + (req.id || ''), 'secondary', 'Admin'); }

  // Add JSON doc
  $('#addJsonForm').on('submit', function (e) {
    e.preventDefault();
    const source_name = $('#json_source_name').val().trim();
    const text = $('#json_text').val().trim();
    const metadataRaw = $('#json_metadata').val().trim();
    if (!source_name || !text) {
      $('#addJsonResponse').html('<div class="alert alert-warning small-muted">Source name and text are required.</div>');
      return;
    }
    let metadata = {};
    if (metadataRaw) {
      try { metadata = JSON.parse(metadataRaw); } catch (e) {
        $('#addJsonResponse').html('<div class="alert alert-danger small-muted">Metadata must be valid JSON.</div>');
        return;
      }
    }
    const payload = { source_name, text, metadata };
    $.ajax({
      url: BASE_API_URL + '/add',
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-API-Key': $('#apiKeyInput').val().trim() || '' },
      data: JSON.stringify(payload),
      success: function (data) { $('#addJsonResponse').html('<div class="alert alert-success small-muted">Added. Response: <pre>' + escapeHtml(JSON.stringify(data, null, 2)) + '</pre></div>'); },
      error: function (xhr) { $('#addJsonResponse').html('<div class="alert alert-danger small-muted">Error: ' + escapeHtml(xhr.responseText || xhr.statusText) + '</div>'); }
    });
  });

  // Upload file multipart
  $('#uploadFileForm').on('submit', function (e) {
    e.preventDefault();
    const fileInput = document.getElementById('upload_file_input');
    if (!fileInput.files || !fileInput.files.length) {
      $('#uploadResponse').html('<div class="alert alert-warning small-muted">Select a file first.</div>');
      return;
    }
    const fd = new FormData();
    fd.append('file', fileInput.files[0]);
    fd.append('department', $('#upload_department').val() || '');
    fd.append('sensitivity', $('#upload_sensitivity').val() || 'public');
    fd.append('tags', $('#upload_tags').val() || '');
    fd.append('public_summary', $('#upload_public_summary').val() || '');
    fd.append('owner_id', $('#upload_owner_id').val() || '');

    $.ajax({
      url: BASE_API_URL + '/add-file',
      method: 'POST',
      headers: { 'X-API-Key': $('#apiKeyInput').val().trim() || '' },
      data: fd,
      processData: false,
      contentType: false,
      success: function (data) { $('#uploadResponse').html('<div class="alert alert-success small-muted">Uploaded. Response: <pre>' + escapeHtml(JSON.stringify(data, null, 2)) + '</pre></div>'); },
      error: function (xhr) { $('#uploadResponse').html('<div class="alert alert-danger small-muted">Error: ' + escapeHtml(xhr.responseText || xhr.statusText) + '</div>'); }
    });
  });

  // Update metadata
  $('#updateMetadataForm').on('submit', function (e) {
    e.preventDefault();
    let ids = [];
    try { ids = JSON.parse($('#update_ids').val()); } catch (e) {
      $('#updateMetadataResponse').html('<div class="alert alert-warning small-muted">IDs must be a JSON array.</div>');
      return;
    }
    let metadata = {};
    try { metadata = JSON.parse($('#update_metadata').val()); } catch (e) {
      $('#updateMetadataResponse').html('<div class="alert alert-warning small-muted">Metadata must be valid JSON.</div>');
      return;
    }
    const payload = { ids, metadata };
    $.ajax({
      url: BASE_API_URL + '/update-metadata',
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-API-Key': $('#apiKeyInput').val().trim() || '' },
      data: JSON.stringify(payload),
      success: function (data) { $('#updateMetadataResponse').html('<div class="alert alert-success small-muted">Updated. Response: <pre>' + escapeHtml(JSON.stringify(data, null, 2)) + '</pre></div>'); },
      error: function (xhr) { $('#updateMetadataResponse').html('<div class="alert alert-danger small-muted">Error: ' + escapeHtml(xhr.responseText || xhr.statusText) + '</div>'); }
    });
  });

  // Composer attach file
  $('#attachBtn').on('click', function () {
    const $fi = $('<input type="file">');
    $fi.on('change', function () {
      const f = this.files[0];
      if (f) {
        state.selectedFile = f;
        $('#fileSelectedRow').removeClass('d-none');
        $('#selectedFilename').text(f.name + ' (' + Math.round(f.size / 1024) + ' KB)');
      }
    });
    $fi.trigger('click');
  });
  $('#clearFileBtn').on('click', function () { state.selectedFile = null; $('#fileSelectedRow').addClass('d-none'); });

  $('#rememberKeyCheckbox').on('change', function () { maybeSaveApiKey(); });
  $('#rememberKeyBtn').on('click', function () { const current = $('#rememberKeyCheckbox').is(':checked'); $('#rememberKeyCheckbox').prop('checked', !current).trigger('change'); });

  $('#topKInput').on('change input', function () { $('#displayTopK').text($(this).val()); });
  $('#useLlmSwitch').on('change', function () { $('#displayUseLlm').text($(this).is(':checked') ? 'on' : 'off'); });

  $('#btnClearHistory').on('click', function () { if (!confirm('Clear chat history locally?')) return; state.history = []; persistHistory(); $('#messages').empty(); updateSessionUI(); });

  $('#apiKeyInput').on('change paste keyup', function () { state.apiKey = $(this).val().trim(); maybeSaveApiKey(); });
  $('#roleSelect').on('change', function () { state.role = $(this).val(); });

  $(window).on('resize', function () { $('#messages').stop().animate({ scrollTop: $('#messages')[0].scrollHeight }, 150); });

  // dev helpers: example responses
  window._EXAMPLE_RESPONSES = {
    employee_filtered_example: {
      answer: "I found restricted documents that are not visible to your role.",
      retrieved: [],
      context: "",
      filtered_out_count: 1,
      public_summaries: ["Bonus is prorated..."]
    },
    hr_example: {
      answer: "Bonus is calculated using a prorated formula based on months worked in the financial year.",
      retrieved: [{ id: "hr_leave_policy.txt_1", text: "The bonus policy: ...", metadata: { department: "HR" } }],
      context: "HR policy context here"
    }
  };
  $('<div class="position-fixed" style="left:1rem; bottom:1rem; z-index:1000;">' +
    '<button id="simEmp" class="btn btn-sm btn-light me-1">Sim Emp Filtered</button>' +
    '<button id="simHr" class="btn btn-sm btn-light">Sim HR</button>' +
    '</div>').appendTo('body');
  $('#simEmp').on('click', function () { renderMessage({ role: 'bot', answer: window._EXAMPLE_RESPONSES.employee_filtered_example.answer, retrieved: [], filtered_out_count: 1, public_summaries: window._EXAMPLE_RESPONSES.employee_filtered_example.public_summaries, ts: new Date().toLocaleString() }); });
  $('#simHr').on('click', function () { renderMessage({ role: 'bot', answer: window._EXAMPLE_RESPONSES.hr_example.answer, retrieved: window._EXAMPLE_RESPONSES.hr_example.retrieved, context: window._EXAMPLE_RESPONSES.hr_example.context, ts: new Date().toLocaleString() }); });

  updateSessionUI();
});

