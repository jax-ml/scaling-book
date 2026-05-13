// Tufte-style margin sidenotes for d-footnote.
//
// On wide viewports (>= 1000px, where the d-article right gutter exists),
// each <d-footnote> gets a sidenote in the right margin, vertically aligned
// with the text line containing its marker. The native d-hover-box popup is
// suppressed; hovering either the in-text marker or the sidenote highlights
// the pair. On narrower viewports the sidenotes are removed and the popup is
// restored.

(function () {
  "use strict";

  const WIDE_QUERY = "(min-width: 1000px)";
  const SIDENOTE_GAP = 12; // min vertical gap (px) between stacked sidenotes
  const CONTAINER_CLASS = "tufte-sidenotes";
  const NOTE_CLASS = "tufte-sidenote";
  const HOVER_CLASS = "tufte-hover";
  const PREF_KEY = "tufte-sidenotes-enabled";

  let container = null;
  let pairs = []; // [{fn, note, off}]
  let raf = null;
  let resizeObs = null;
  let lastArticleHeight = -1;

  function isWide() {
    return window.matchMedia(WIDE_QUERY).matches;
  }

  function isEnabled() {
    try {
      return localStorage.getItem(PREF_KEY) === "1";
    } catch (e) {
      return false;
    }
  }

  function setEnabled(on) {
    try {
      localStorage.setItem(PREF_KEY, on ? "1" : "0");
    } catch (e) {}
    document.documentElement.classList.toggle("tufte-mode", on);
  }

  function getArticle() {
    return document.querySelector("d-article");
  }

  function getFootnotes(article) {
    return Array.from(article.querySelectorAll("d-footnote"));
  }

  // d-hover-box.show() force-sets style.display = "block", so simply hiding
  // the box doesn't stick. Instead, neuter show() while sidenotes are active
  // and restore the original when they aren't.
  function setPopupSuppressed(footnotes, suppressed) {
    for (const fn of footnotes) {
      const root = fn.shadowRoot;
      if (!root) continue;
      const hover = root.querySelector("d-hover-box");
      if (!hover) continue;
      if (suppressed) {
        if (!hover.__sidenoteShow) {
          hover.__sidenoteShow = hover.show;
          hover.show = function () {};
          // Inject a hover style into the shadow root so we can flash the
          // marker number from outside (regular CSS can't pierce shadow DOM).
          const style = document.createElement("style");
          style.setAttribute("data-sidenote", "");
          style.textContent =
            "sup span{cursor:pointer;}" +
            "sup span." + HOVER_CLASS + "{color:#ff8c00;}";
          root.appendChild(style);
        }
        hover.style.display = "none";
      } else if (hover.__sidenoteShow) {
        hover.show = hover.__sidenoteShow;
        delete hover.__sidenoteShow;
        hover.style.display = "";
      }
    }
  }

  function getFootnoteNumber(fn, index) {
    const root = fn.shadowRoot;
    if (root) {
      const span = root.querySelector("sup span");
      if (span && span.textContent) return span.textContent;
    }
    return String(index + 1);
  }

  // A footnote inside a collapsed <details> (or otherwise hidden) should not
  // produce a floating sidenote.
  function isVisible(el) {
    if (!el.isConnected) return false;
    if (el.closest("details:not([open])")) return false;
    const r = el.getBoundingClientRect();
    return r.width > 0 || r.height > 0;
  }

  // Box of the text line containing the marker, relative to the article.
  // d-footnote is inline; its box height equals the line-height of the
  // surrounding text (the sup is position:relative so it doesn't grow the
  // box), so rect.top/height are the line top/height.
  function lineBox(fn, articleTop) {
    const r = fn.getBoundingClientRect();
    return { top: r.top - articleTop, height: r.height };
  }

  function setHoverPair(fn, note, on) {
    note.classList.toggle(HOVER_CLASS, on);
    fn.classList.toggle(HOVER_CLASS, on);
    // The visible marker number is in shadow DOM; toggle a class on it too so
    // the highlight can reach it via :host().
    const root = fn.shadowRoot;
    if (root) {
      const span = root.querySelector("sup span");
      if (span) span.classList.toggle(HOVER_CLASS, on);
    }
  }

  function bindHoverPair(fn, note) {
    const on = () => setHoverPair(fn, note, true);
    const off = () => setHoverPair(fn, note, false);
    const jump = (e) => {
      // Don't hijack clicks on links inside the sidenote.
      if (e.target.closest("a")) return;
      fn.scrollIntoView({ behavior: "smooth", block: "center" });
      // Briefly flash the marker so it's findable once the mouse leaves the
      // sidenote (which would otherwise immediately drop the highlight).
      setHoverPair(fn, note, true);
      setTimeout(off, 1200);
    };
    fn.addEventListener("mouseenter", on);
    fn.addEventListener("mouseleave", off);
    fn.addEventListener("click", jump);
    note.addEventListener("mouseenter", on);
    note.addEventListener("mouseleave", off);
    note.addEventListener("click", jump);
    return () => {
      fn.removeEventListener("mouseenter", on);
      fn.removeEventListener("mouseleave", off);
      fn.removeEventListener("click", jump);
      note.removeEventListener("mouseenter", on);
      note.removeEventListener("mouseleave", off);
      note.removeEventListener("click", jump);
      setHoverPair(fn, note, false);
    };
  }

  function clearSidenotes(article, footnotes) {
    for (const p of pairs) p.off();
    pairs = [];
    if (container && container.parentNode) container.parentNode.removeChild(container);
    container = null;
    setPopupSuppressed(footnotes, false);
    if (article) article.classList.remove("has-sidenotes");
  }

  function buildSidenotes() {
    const article = getArticle();
    if (!article) return;
    const footnotes = getFootnotes(article);

    if (!isWide() || !isEnabled() || footnotes.length === 0) {
      clearSidenotes(article, footnotes);
      return;
    }

    // Reset and rebuild from scratch.
    for (const p of pairs) p.off();
    pairs = [];
    if (container && container.parentNode) container.parentNode.removeChild(container);

    container = document.createElement("div");
    container.className = CONTAINER_CLASS;
    container.setAttribute("aria-hidden", "true");
    article.appendChild(container);
    article.classList.add("has-sidenotes");

    const articleTop = article.getBoundingClientRect().top;
    let prevBottom = -Infinity;

    footnotes.forEach((fn, i) => {
      if (!isVisible(fn)) return;

      const note = document.createElement("div");
      note.className = NOTE_CLASS;

      const num = document.createElement("sup");
      num.className = NOTE_CLASS + "-number";
      num.textContent = getFootnoteNumber(fn, i);
      note.appendChild(num);
      note.appendChild(document.createTextNode(" "));

      const body = document.createElement("span");
      body.className = NOTE_CLASS + "-body";
      for (const child of Array.from(fn.childNodes)) {
        body.appendChild(child.cloneNode(true));
      }
      note.appendChild(body);

      container.appendChild(note);

      // Center the sidenote's first line on the body text line containing the
      // marker, so the smaller sidenote text doesn't float above the body
      // baseline (which is what raw top-to-top alignment gives).
      const lb = lineBox(fn, articleTop);
      const noteLH = parseFloat(getComputedStyle(note).lineHeight) || lb.height;
      const wantTop = lb.top + (lb.height - noteLH) / 2;
      const top = Math.max(wantTop, prevBottom + SIDENOTE_GAP);
      note.style.top = top + "px";
      prevBottom = top + note.getBoundingClientRect().height;

      const off = bindHoverPair(fn, note);
      pairs.push({ fn, note, off });
    });

    setPopupSuppressed(footnotes, true);
  }

  function scheduleBuild() {
    if (raf) cancelAnimationFrame(raf);
    raf = requestAnimationFrame(() => {
      raf = null;
      buildSidenotes();
    });
  }

  // Floating settings control (gear button + small popover with a checkbox).
  // Only shown when the viewport is wide enough for sidenotes to be possible.
  function buildSettings() {
    const wrap = document.createElement("div");
    wrap.className = "tufte-settings";

    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "tufte-settings-btn";
    btn.title = "Display settings";
    btn.setAttribute("aria-haspopup", "true");
    btn.setAttribute("aria-expanded", "false");
    btn.innerHTML =
      '<svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true">' +
      '<path fill="currentColor" d="M19.14 12.94a7.07 7.07 0 0 0 0-1.88l2.03-1.58a.5.5 0 0 0 ' +
      ".12-.64l-1.92-3.32a.5.5 0 0 0-.6-.22l-2.39.96a7.03 7.03 0 0 0-1.62-.94l-.36-2.54a.5.5 " +
      "0 0 0-.5-.42h-3.84a.5.5 0 0 0-.5.42l-.36 2.54c-.58.24-1.12.55-1.62.94l-2.39-.96a.5.5 " +
      "0 0 0-.6.22L2.71 8.84a.5.5 0 0 0 .12.64l2.03 1.58a7.07 7.07 0 0 0 0 1.88l-2.03 " +
      "1.58a.5.5 0 0 0-.12.64l1.92 3.32c.13.22.39.31.6.22l2.39-.96c.5.39 1.04.7 " +
      "1.62.94l.36 2.54c.04.24.25.42.5.42h3.84c.25 0 .46-.18.5-.42l.36-2.54c.58-.24 " +
      "1.12-.55 1.62-.94l2.39.96c.21.09.47 0 .6-.22l1.92-3.32a.5.5 0 0 0-.12-.64l-2.03-1.58zM12 " +
      '15.5A3.5 3.5 0 1 1 12 8.5a3.5 3.5 0 0 1 0 7z"/></svg>';
    wrap.appendChild(btn);

    const panel = document.createElement("div");
    panel.className = "tufte-settings-panel";
    panel.hidden = true;
    const id = "tufte-sidenotes-toggle";
    panel.innerHTML =
      '<label for="' + id + '">' +
      '<input type="checkbox" id="' + id + '"> Tufte footnotes' +
      "</label>";
    wrap.appendChild(panel);

    const cb = panel.querySelector("input");
    cb.checked = isEnabled();
    cb.addEventListener("change", () => {
      setEnabled(cb.checked);
      scheduleBuild();
    });

    btn.addEventListener("click", (e) => {
      e.stopPropagation();
      panel.hidden = !panel.hidden;
      btn.setAttribute("aria-expanded", String(!panel.hidden));
    });
    document.addEventListener("click", (e) => {
      if (!panel.hidden && !wrap.contains(e.target)) {
        panel.hidden = true;
        btn.setAttribute("aria-expanded", "false");
      }
    });

    document.body.appendChild(wrap);
  }

  function init() {
    const article = getArticle();
    if (!article) return;

    // Sync class in case the early head script and the deferred script see
    // different localStorage state (e.g. cleared mid-session).
    document.documentElement.classList.toggle("tufte-mode", isEnabled());
    buildSettings();
    scheduleBuild();
    window.addEventListener("resize", scheduleBuild);
    window.addEventListener("load", scheduleBuild);

    // Images, KaTeX, fonts, and figures load asynchronously and shift layout.
    // Re-layout whenever the article's box height changes. The sidenote
    // container is abspos so it doesn't affect the article's box, but guard
    // against self-triggering anyway.
    if (window.ResizeObserver) {
      resizeObs = new ResizeObserver((entries) => {
        const h = entries[0].contentRect.height;
        if (Math.abs(h - lastArticleHeight) < 1) return;
        lastArticleHeight = h;
        scheduleBuild();
      });
      resizeObs.observe(article);
    }

    // Re-layout when collapsible answer blocks open/close.
    article.addEventListener(
      "toggle",
      (e) => {
        if (e.target && e.target.tagName === "DETAILS") scheduleBuild();
      },
      true
    );
  }

  if (window.customElements && customElements.whenDefined) {
    Promise.all([
      customElements.whenDefined("d-footnote"),
      new Promise((resolve) => {
        if (document.readyState !== "loading") resolve();
        else document.addEventListener("DOMContentLoaded", resolve, { once: true });
      }),
    ]).then(init);
  } else {
    document.addEventListener("DOMContentLoaded", init);
  }
})();
