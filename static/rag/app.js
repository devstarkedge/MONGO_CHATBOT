/* Mongo RAG UI. Plain JavaScript, no libraries. Talks to /api/ask/, /api/search/, /api/ingest/, /api/health/.
   All dynamic text is inserted with textContent / text nodes, so database or LLM output can never inject HTML. */
(() => {
    "use strict";

    const $ = (sel, root = document) => root.querySelector(sel);
    const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

    // ------------------------------------------------------------------ helpers
    function h(tag, attrs, ...kids) {
        const el = document.createElement(tag);
        for (const [key, value] of Object.entries(attrs || {})) {
            if (value === null || value === undefined || value === false) continue;
            if (key === "class") el.className = value;
            else if (key.startsWith("on")) el.addEventListener(key.slice(2), value);
            else el.setAttribute(key, value === true ? "" : value);
        }
        for (const kid of kids.flat()) {
            if (kid === null || kid === undefined || kid === false) continue;
            el.append(kid);
        }
        return el;
    }

    const MARK = '<svg viewBox="0 0 24 24" width="16" height="16" aria-hidden="true"><path d="M12 2.5l8.2 4.75v9.5L12 21.5l-8.2-4.75v-9.5z" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linejoin="round"/><circle cx="12" cy="12" r="3.2" fill="currentColor"/></svg>';
    const COPY = '<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="9" y="9" width="11" height="11" rx="2"/><path d="M5 15V6a2 2 0 0 1 2-2h9"/></svg>';
    const EDIT = '<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 20h9"/><path d="M16.5 3.5a2.12 2.12 0 0 1 3 3L7 19l-4 1 1-4Z"/></svg>';
    function staticSvg(markup, cls) {
        const span = h("span", { class: cls });
        span.innerHTML = markup; // constant markup written in this file, never user data
        return span;
    }

    async function api(path, options) {
        const response = await fetch(path, options);
        let data = {};
        try { data = await response.json(); } catch (e) { /* non-JSON error page */ }
        return { ok: response.ok, status: response.status, data };
    }

    // ---------------------------- CSRF helpers (Django)
    function getCookie(name) {
        const value = `; ${document.cookie}`;
        const parts = value.split(`; ${name}=`);
        if (parts.length === 2) return parts.pop().split(";").shift();
    }
    function csrfHeaders() {
        const token = getCookie("csrftoken");
        return token ? { "X-CSRFToken": token } : {};
    }

    const hue = (name) => { let n = 0; for (const c of String(name)) n = (n * 31 + c.charCodeAt(0)) % 360; return n; };
    const shortId = (id) => (String(id).length > 10 ? String(id).slice(0, 8) + "…" : String(id));
    const when = (iso) => { try { return new Date(iso).toLocaleString(); } catch (e) { return iso || ""; } };

    // ---------------------------------------------- tiny, safe markdown renderer
    function inline(text) {
        const frag = document.createDocumentFragment();
        for (const part of text.split(/(\*\*[^*\n]+\*\*|`[^`\n]+`)/g)) {
            if (!part) continue;
            if (part.length > 4 && part.startsWith("**") && part.endsWith("**")) frag.append(h("strong", {}, part.slice(2, -2)));
            else if (part.length > 2 && part.startsWith("`") && part.endsWith("`")) frag.append(h("code", {}, part.slice(1, -1)));
            else frag.append(part);
        }
        return frag;
    }

    const isBullet = (l) => /^\s*[-*•]\s+/.test(l);
    const isNumber = (l) => /^\s*\d+[.)]\s+/.test(l);
    const isTableSep = (l) => /^\s*\|?(\s*:?-{2,}:?\s*\|?)+\s*$/.test(l || "");
    const splitRow = (l) => l.trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map((c) => c.trim());
    const startsBlock = (l, next) => /^```/.test(l.trim()) || isBullet(l) || isNumber(l) || /^#{1,6}\s+/.test(l)
        || (/^\s*\|/.test(l) && isTableSep(next));

    function renderMarkdown(source) {
        const root = h("div", { class: "md" });
        const lines = String(source).replace(/\r\n?/g, "\n").split("\n");
        let i = 0;
        while (i < lines.length) {
            const line = lines[i];
            if (!line.trim()) { i++; continue; }

            if (/^```/.test(line.trim())) {
                const buf = [];
                for (i++; i < lines.length && !/^```/.test(lines[i].trim()); i++) buf.push(lines[i]);
                i++;
                root.append(h("pre", {}, h("code", {}, buf.join("\n"))));
                continue;
            }
            if (/^\s*\|/.test(line) && isTableSep(lines[i + 1])) {
                const head = splitRow(line);
                const body = [];
                for (i += 2; i < lines.length && /^\s*\|/.test(lines[i]); i++) body.push(splitRow(lines[i]));
                root.append(h("div", { class: "tablewrap" }, h("table", {},
                    h("thead", {}, h("tr", {}, head.map((c) => h("th", {}, inline(c))))),
                    h("tbody", {}, body.map((r) => h("tr", {}, r.map((c) => h("td", {}, inline(c)))))))));
                continue;
            }
            if (isBullet(line) || isNumber(line)) {
                const ordered = isNumber(line);
                const items = [];
                for (; i < lines.length && (ordered ? isNumber(lines[i]) : isBullet(lines[i])); i++) {
                    items.push(h("li", {}, inline(lines[i].replace(ordered ? /^\s*\d+[.)]\s+/ : /^\s*[-*•]\s+/, ""))));
                }
                root.append(h(ordered ? "ol" : "ul", {}, items));
                continue;
            }
            if (/^#{1,6}\s+/.test(line)) {
                root.append(h("p", {}, h("strong", {}, inline(line.replace(/^#{1,6}\s+/, "")))));
                i++;
                continue;
            }
            const para = [];
            while (i < lines.length && lines[i].trim() && !(para.length && startsBlock(lines[i], lines[i + 1]))) {
                para.push(lines[i]);
                i++;
            }
            const p = h("p", {});
            para.forEach((l, n) => { if (n) p.append(h("br")); p.append(inline(l)); });
            root.append(p);
        }
        return root;
    }

    // ------------------------------------------------------------------- tabs
    const views = { chat: $("#view-chat"), search: $("#view-search"), data: $("#view-data") };
    function showView(name) {
        if (!views[name]) name = "chat";
        Object.entries(views).forEach(([key, el]) => { el.hidden = key !== name; });
        $$(".tab").forEach((tab) => {
            if (tab.dataset.view === name) tab.setAttribute("aria-current", "page");
            else tab.removeAttribute("aria-current");
        });
        if (location.hash !== "#" + name) history.replaceState(null, "", "#" + name);
        if (name === "data") loadHealth();
        if (name === "chat") $("#question").focus();
        if (name === "search") $("#searchInput").focus();
    }
    $$(".tab").forEach((tab) => tab.addEventListener("click", () => showView(tab.dataset.view)));
    $(".brand").addEventListener("click", (e) => { e.preventDefault(); showView("chat"); });
    $("#statusPill").addEventListener("click", () => showView("data"));

    // ----------------------------------------------------------------- status
    function setStatus(state, text) {
        $("#statusPill").dataset.state = state;
        $("#statusText").textContent = text;
    }

    async function loadHealth() {
        let result;
        try { result = await api("/api/health/"); } catch (e) {
            setStatus("bad", "Server unreachable");
            renderStats(null);
            return;
        }
        const d = result.data;
        const llm = d.llm || {};
        if (llm.provider === "ollama" && llm.reachable === false) setStatus("warn", "Ollama is not running");
        else if (llm.provider === "ollama" && llm.model_available === false) setStatus("warn", "Run: ollama pull " + llm.model);
        else if (d.index && d.index.ready) setStatus(d.status === "ok" ? "ok" : "warn", d.embeddings + " chunks indexed");
        else setStatus("bad", (d.index && d.index.message) || d.error || "Index not built yet");
        renderStats(result.ok ? d : null);
    }

    function stat(label, value, sub, small) {
        return h("div", { class: "card stat" },
            h("div", { class: "k" }, label), h("div", { class: "v" + (small ? " small" : "") }, value), sub ? h("div", { class: "s" }, sub) : null);
    }

    function llmNote(llm) {
        if (!llm) return null;
        if (llm.provider !== "ollama") return llm.provider;
        if (!llm.reachable) return "Ollama is not running (start the app or: ollama serve)";
        return llm.model_available ? "Ollama · running locally" : "Ollama · model not installed: ollama pull " + llm.model;
    }

    function renderStats(d) {
        const box = $("#stats");
        box.replaceChildren();
        if (!d) { box.append(stat("Status", "Unavailable", "Could not read /api/health/")); return; }
        const idx = d.index || {};
        box.append(
            stat("Search index", idx.ready ? "Ready" : "Not built", idx.ready ? `${idx.vectors} vectors · built ${when(idx.built_at)}` : (idx.message || "")),
            stat("Embedded chunks", String(d.embeddings), "stored in MongoDB"),
            stat("MongoDB", d.mongo ? "Connected" : "Unreachable"),
            stat("Embedding model", d.embedding_model, null, true),
            stat("Language model", d.llm_model, llmNote(d.llm), true),
            stat("Hybrid search", d.hybrid_search ? "On" : "Off", "vectors + keywords"),
            stat("Answer cache", `${d.cache.entries} entries`, `${d.cache.hits} hits · ${d.cache.misses} misses`));
    }

    // ------------------------------------------------------------------- chat
    const MODES = { fast: { top_k: 3, hops: 0 }, indepth: { top_k: 6, hops: 1 }, holistic: { top_k: 10, hops: 2 } };
    let mode = "indepth";
    let busy = false;

    const chat = views.chat;
    const thread = $("#thread");
    const inner = $("#threadInner");
    const input = $("#question");
    const sendBtn = $("#send");
    const SESSION_STORAGE_KEY = "mongo-rag-chat-session";
    const newSessionId = () => (crypto.randomUUID ? crypto.randomUUID() :
        "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (c) => {
            const r = Math.random() * 16 | 0;
            return (c === "x" ? r : (r & 0x3 | 0x8)).toString(16);
        }));
    let chatSessionId;
    try {
        chatSessionId = sessionStorage.getItem(SESSION_STORAGE_KEY);
        if (!chatSessionId) {
            chatSessionId = newSessionId();
            sessionStorage.setItem(SESSION_STORAGE_KEY, chatSessionId);
        }
    } catch (e) {
        chatSessionId = newSessionId();
    }
    thread.classList.add("show-sources");

    // Keeps a parallel list of rendered user turns so an edit can trim the thread.
    // Each entry: { wrap, bubble, text, botNode }
    const userTurns = [];

    $$(".mode").forEach((btn) => btn.addEventListener("click", () => {
        mode = btn.dataset.mode;
        $$(".mode").forEach((b) => b.setAttribute("aria-checked", String(b === btn)));
        input.focus();
    }));
    [["#toggleSources", "show-sources"], ["#toggleTiming", "show-timing"]].forEach(([sel, cls]) => {
        const btn = $(sel);
        btn.addEventListener("click", () => {
            const on = thread.classList.toggle(cls);
            btn.setAttribute("aria-pressed", String(on));
        });
    });

    function scrollEnd() { thread.scrollTop = thread.scrollHeight; }
    function setEmpty(empty) { chat.classList.toggle("is-empty", empty); }

    function updateComposer() {
        const length = input.value.length;
        sendBtn.disabled = busy || !input.value.trim();
        const count = $("#count");
        count.hidden = length < 350;
        count.textContent = `${length}/500`;
        input.style.height = "auto";
        input.style.height = Math.min(input.scrollHeight, 160) + "px";
    }

    function addUser(text) {
        setEmpty(false);
        const bubble = h("div", { class: "msg user" }, text);
        const editBtn = h("button", {
            type: "button",
            class: "msg-edit-btn",
            "aria-label": "Edit message",
            title: "Edit message",
        }, staticSvg(EDIT), "Edit");
        const actions = h("div", { class: "msg-edit-actions" }, editBtn);
        const wrap = h("div", { class: "msg user-wrap" }, bubble, actions);
        inner.append(wrap);
        scrollEnd();

        const turn = { wrap, bubble, text, botNode: null };
        userTurns.push(turn);
        editBtn.addEventListener("click", () => beginEditMessage(turn));
        return wrap;
    }

    function beginEditMessage(turn) {
        if (busy) return;
        const { bubble } = turn;
        if (bubble.classList.contains("editing")) return;

        const original = turn.text;
        bubble.classList.add("editing");
        bubble.replaceChildren();

        const ta = h("textarea", { rows: "2", maxlength: "500" });
        ta.value = original;
        const save = h("button", { type: "button", class: "primary" }, "Send");
        const cancel = h("button", { type: "button" }, "Cancel");
        bubble.append(ta, h("div", { class: "edit-bar" }, cancel, save));
        ta.focus();
        ta.setSelectionRange(ta.value.length, ta.value.length);

        const autoGrow = () => { ta.style.height = "auto"; ta.style.height = Math.min(ta.scrollHeight, 200) + "px"; };
        autoGrow();
        ta.addEventListener("input", autoGrow);

        const close = (restore) => {
            bubble.classList.remove("editing");
            bubble.replaceChildren();
            if (restore) bubble.append(turn.text);
        };

        cancel.addEventListener("click", () => close(true));
        ta.addEventListener("keydown", (e) => {
            if (e.key === "Escape") { e.preventDefault(); close(true); }
            if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); save.click(); }
        });

        save.addEventListener("click", async () => {
            const next = ta.value.trim();
            if (!next) { ta.focus(); return; }
            if (next === original) { close(true); return; }
            close(false);
            bubble.append(next);
            turn.text = next;
            // Drop everything after this turn (client-side "edit & regenerate")
            const idx = userTurns.indexOf(turn);
            const removed = userTurns.splice(idx + 1);
            removed.forEach((t) => {
                if (t.botNode) t.botNode.remove();
                t.wrap.remove();
            });
            // Clear the bot reply attached to this turn (we are regenerating it)
            if (turn.botNode) { turn.botNode.remove(); turn.botNode = null; }
            await ask(next, { skipUserRender: true });
        });
    }

    function sourceChip(s) {
        const chip = h("span", {
            class: "src" + (s.related ? " related" : ""),
            title: `${s.collection} · ${s.document_id}` + (s.via ? ` (linked via ${s.via.field})` : ""),
        }, h("b", {}, s.collection), s.name ? h("span", {}, s.name) : null, h("code", {}, shortId(s.document_id)),
            s.related ? h("span", { class: "tag" }, "linked") : null);
        chip.style.setProperty("--h", hue(s.collection));
        return chip;
    }

    function addBot(parts, isError) {
        const bubble = h("div", { class: "bubble" }, parts.body);
        if (parts.sources && parts.sources.length) {
            bubble.append(h("div", { class: "sources" }, h("span", { class: "src-label" }, "Sources"), parts.sources.map(sourceChip)));
        }
        if (parts.timing) bubble.append(h("div", { class: "timing" }, parts.timing));
        if (parts.copyText) {
            const btn = h("button", { type: "button", class: "icon-btn", "aria-label": "Copy answer" }, staticSvg(COPY), "Copy");
            btn.addEventListener("click", async () => {
                try { await navigator.clipboard.writeText(parts.copyText); btn.lastChild.textContent = "Copied"; }
                catch (e) { btn.lastChild.textContent = "Press Ctrl+C"; }
                setTimeout(() => { btn.lastChild.textContent = "Copy"; }, 1500);
            });
            bubble.append(h("div", { class: "actions" }, btn));
        }
        const msg = h("div", { class: "msg bot" + (isError ? " error" : "") }, staticSvg(MARK, "avatar"), bubble);
        inner.append(msg);
        scrollEnd();
        return msg;
    }

    function addTyping() {
        const msg = addBot({ body: h("div", { class: "typing", role: "status", "aria-label": "Looking up an answer" }, h("i"), h("i"), h("i")) });
        return msg;
    }

    function timingText(data) {
        const bits = Object.entries(data.timings_ms || {}).map(([k, v]) => `${k} ${v} ms`);
        if (data.mode === "structured") {
            const cond = data.conditions && data.conditions !== "none" ? " · " + data.conditions : "";
            bits.unshift(`exact query: ${data.total_matching} in ${data.collection}${cond}`);
        } else if (data.records_used) bits.unshift(`${data.records_used} records${data.list_mode ? " (wide search)" : ""}`);
        if (data.degraded) bits.push("plain result: language model unavailable");
        if (data.cached) bits.push("from cache");
        return bits.join(" · ");
    }

    async function ask(question, opts = {}) {
        question = (question || "").trim();
        if (!question || busy) return;
        busy = true;
        if (!opts.skipUserRender) addUser(question);
        input.value = "";
        updateComposer();
        const typing = addTyping();
        try {
            const { ok, status, data } = await api("/api/ask/", {
                method: "POST",
                headers: { "Content-Type": "application/json", ...csrfHeaders() },
                body: JSON.stringify({ question, session_id: chatSessionId, ...MODES[mode] }),
            });
            typing.remove();
            if (ok) {
                const botMsg = addBot({ body: renderMarkdown(data.answer), sources: data.sources, timing: timingText(data), copyText: data.answer });
                const lastTurn = userTurns[userTurns.length - 1];
                if (lastTurn) lastTurn.botNode = botMsg;
                loadChatSessions();
            }
            else addBot({ body: data.error || `The server returned an error (${status}).` }, true);
        } catch (e) {
            typing.remove();
            addBot({ body: "Couldn't reach the server. Check that it is running and try again." }, true);
        } finally {
            busy = false;
            updateComposer();
            input.focus();
        }
    }

    $("#composer").addEventListener("submit", (e) => { e.preventDefault(); ask(input.value); });
    input.addEventListener("input", updateComposer);
    input.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); ask(input.value); }
    });
    $$(".chip").forEach((chip) => chip.addEventListener("click", () => ask(chip.textContent)));
    const chatSidebar = $("#chatSidebar");
    const chatHistoryList = $("#chatHistoryList");

    $("#sidebarToggle").addEventListener("click", () => {
        chat.classList.toggle("sidebar-collapsed");
    });
    $("#sidebarOpen").addEventListener("click", () => {
        chat.classList.remove("sidebar-collapsed");
    });

    async function loadChatSessions() {
        try {
            const { ok, data } = await api("/api/chats/");
            if (!ok || !data.sessions) return;
            chatHistoryList.replaceChildren();
            data.sessions.forEach((sess) => {
                const active = sess.session_id === chatSessionId;
                const titleSpan = h("span", { class: "chat-history-title" }, sess.title);

                const editBtn = h("button", {
                    class: "chat-history-edit",
                    title: "Rename chat",
                    "aria-label": "Rename chat",
                    onclick: (e) => { e.stopPropagation(); beginRename(sess, titleSpan, editBtn); }
                }, "✎");

                const delBtn = h("button", {
                    class: "chat-history-delete",
                    title: "Delete chat",
                    onclick: (e) => { e.stopPropagation(); deleteSession(sess.session_id); }
                }, "✕");

                const item = h("div", {
                    class: "chat-history-item" + (active ? " active" : ""),
                    title: sess.title,
                    onclick: () => switchSession(sess.session_id)
                }, titleSpan, editBtn, delBtn);

                chatHistoryList.append(item);
            });
        } catch (e) { }
    }

    function beginRename(sess, titleSpan, editBtn) {
        if (titleSpan.dataset.editing === "1") return;
        titleSpan.dataset.editing = "1";
        const original = sess.title;
        const inp = h("input", { class: "chat-history-title-input", maxlength: "120" });
        inp.value = original;
        titleSpan.replaceWith(inp);
        inp.focus();
        inp.select();

        const finish = async (commit) => {
            titleSpan.dataset.editing = "";
            if (!commit || !inp.value.trim() || inp.value.trim() === original) {
                if (inp.parentNode) inp.replaceWith(titleSpan);
                return;
            }
            const next = inp.value.trim();
            titleSpan.textContent = next;
            sess.title = next;
            if (inp.parentNode) inp.replaceWith(titleSpan);
            try {
                const { ok } = await api(`/api/chats/${sess.session_id}/rename/`, {
                    method: "POST",
                    headers: { "Content-Type": "application/json", ...csrfHeaders() },
                    body: JSON.stringify({ title: next }),
                });
                if (!ok) {
                    // Roll back optimistic update if the server rejected it
                    titleSpan.textContent = original;
                    sess.title = original;
                }
            } catch (e) {
                titleSpan.textContent = original;
                sess.title = original;
            }
        };

        inp.addEventListener("keydown", (e) => {
            if (e.key === "Enter") { e.preventDefault(); finish(true); }
            if (e.key === "Escape") { e.preventDefault(); finish(false); }
        });
        inp.addEventListener("blur", () => finish(true));
        inp.addEventListener("click", (e) => e.stopPropagation());
    }

    async function switchSession(sid) {
        chatSessionId = sid;
        try { sessionStorage.setItem(SESSION_STORAGE_KEY, sid); } catch (e) { }
        inner.replaceChildren();
        userTurns.length = 0;
        try {
            const { ok, data } = await api(`/api/chats/${sid}/`);
            if (ok && data.turns && data.turns.length > 0) {
                setEmpty(false);
                data.turns.forEach((turn) => {
                    addUser(turn.question);
                    const botMsg = addBot({ body: renderMarkdown(turn.answer) });
                    const lastTurn = userTurns[userTurns.length - 1];
                    if (lastTurn) lastTurn.botNode = botMsg;
                });
            } else {
                setEmpty(true);
            }
        } catch (e) {
            setEmpty(true);
        }
        loadChatSessions();
        input.focus();
    }

    async function deleteSession(sid) {
        try {
            await api(`/api/chats/${sid}/delete/`, {
                method: "POST",
                headers: { ...csrfHeaders() },
            });
            if (chatSessionId === sid) {
                startNewChat();
            } else {
                loadChatSessions();
            }
        } catch (e) { }
    }

    function startNewChat() {
        inner.replaceChildren();
        userTurns.length = 0;
        setEmpty(true);
        chatSessionId = newSessionId();
        try { sessionStorage.setItem(SESSION_STORAGE_KEY, chatSessionId); } catch (e) { }
        loadChatSessions();
        input.focus();
    }

    $("#sidebarNewChat").addEventListener("click", startNewChat);
    $("#newChat").addEventListener("click", startNewChat);

    document.addEventListener("keydown", (e) => {
        const typing = /^(INPUT|TEXTAREA|SELECT)$/.test((document.activeElement || {}).tagName || "");
        if (e.key === "/" && !typing && !chat.hidden) { e.preventDefault(); input.focus(); }
    });

    // ----------------------------------------------------------------- search
    function resultCard(r, related) {
        const card = h("article", { class: "card result" });
        card.style.setProperty("--h", hue(r.collection));
        const score = (label, v) => (v === null || v === undefined ? null : h("span", { class: "score" }, `${label} ${v}`));
        card.append(h("div", { class: "result-head" },
            h("span", { class: "src" }, h("b", {}, r.collection)),
            h("span", { class: "result-name" }, r.name || shortId(r.document_id)),
            h("code", { title: r.document_id }, shortId(r.document_id)),
            related ? h("span", { class: "tag" }, (r.via && r.via.direction === "reverse") ? "points here" : "linked") : null,
            related ? null : h("span", { class: "scores" }, score("match", r.score), score("vector", r.vector_score), score("keyword", r.keyword_score))));
        card.append(h("pre", {}, r.text));
        if ((r.text || "").length > 380) {
            const toggle = h("button", { type: "button", class: "link-btn" }, "Show more");
            toggle.addEventListener("click", () => { const open = card.classList.toggle("open"); toggle.textContent = open ? "Show less" : "Show more"; });
            card.append(toggle);
        }
        if (r.relationships && r.relationships.length) {
            card.append(h("div", { class: "rels" }, r.relationships.map((rel) =>
                h("span", { class: "rel" }, `${rel.field} → ${rel.ref_collection}${rel.ref_name ? ": " + rel.ref_name : ""}`))));
        }
        return card;
    }

    $("#searchForm").addEventListener("submit", async (e) => {
        e.preventDefault();
        const query = $("#searchInput").value.trim();
        const out = $("#searchOut");
        if (!query) return;
        const btn = $("#searchBtn");
        btn.disabled = true;
        out.replaceChildren(h("div", { class: "empty-note" }, "Searching…"));
        try {
            const expand = $("#searchExpand").checked;
            const { ok, status, data } = await api("/api/search/", {
                method: "POST",
                headers: { "Content-Type": "application/json", ...csrfHeaders() },
                body: JSON.stringify({ query, top_k: Number($("#searchK").value), hybrid: $("#searchHybrid").checked, expand }),
            });
            out.replaceChildren();
            if (!ok) { out.append(h("div", { class: "card empty-note" }, data.error || `Error ${status}`)); return; }
            if (!data.results.length) { out.append(h("div", { class: "card empty-note" }, "No matching records.")); return; }
            data.results.forEach((r) => out.append(resultCard(r, false)));
            if (data.related && data.related.length) {
                out.append(h("div", { class: "group-title" }, "Linked records"));
                data.related.forEach((r) => out.append(resultCard(r, true)));
            }
            out.append(h("div", { class: "muted" }, `${data.results.length} result(s) in ${data.timings_ms.total} ms`));
        } catch (err) {
            out.replaceChildren(h("div", { class: "card empty-note" }, "Couldn't reach the server."));
        } finally {
            btn.disabled = false;
        }
    });

    // -------------------------------------------------------------- ingestion
    let pollTimer = null;
    function authHeaders() {
        const token = $("#tokenInput").value.trim();
        return token ? { Authorization: "Bearer " + token } : {};
    }

    function renderJob(job) {
        $("#jobBox").hidden = false;
        const state = $("#jobState");
        state.textContent = job.state;
        state.dataset.state = job.state;
        const bits = [];
        if (job.started_at) bits.push("started " + when(job.started_at));
        if (job.summary) bits.push(`${job.summary.total_vectors} vectors in ${job.summary.seconds}s`);
        $("#jobMeta").textContent = bits.join(" · ");
        const lines = (job.log || []).slice();
        if (job.error) lines.push("ERROR: " + job.error);
        $("#jobLog").textContent = lines.join("\n");
    }

    async function poll() {
        try {
            const { ok, data } = await api("/api/ingest/", { headers: authHeaders() });
            if (!ok) { clearInterval(pollTimer); pollTimer = null; return; }
            renderJob(data);
            if (data.state !== "running") {
                clearInterval(pollTimer);
                pollTimer = null;
                $("#ingestBtn").disabled = false;
                loadHealth();
            }
        } catch (e) { /* keep polling */ }
    }

    $("#ingestBtn").addEventListener("click", async () => {
        const btn = $("#ingestBtn");
        btn.disabled = true;
        try {
            const { ok, status, data } = await api("/api/ingest/", {
                method: "POST",
                headers: { "Content-Type": "application/json", ...authHeaders(), ...csrfHeaders() },
                body: JSON.stringify({ full: $("#fullInput").checked }),
            });
            if (!ok) {
                renderJob({ state: "failed", log: [], error: data.error || `Error ${status}` });
                btn.disabled = false;
                return;
            }
            renderJob(data);
            pollTimer = setInterval(poll, 1500);
        } catch (e) {
            renderJob({ state: "failed", log: [], error: "Couldn't reach the server." });
            btn.disabled = false;
        }
    });

    // ------------------------------------------------------------------- boot
    showView((location.hash || "#chat").slice(1));
    updateComposer();
    loadHealth();
    switchSession(chatSessionId);
})();