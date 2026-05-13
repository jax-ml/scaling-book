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

  let container = null;
  let pairs = []; // [{fn, note, off}]
  let raf = null;
  let resizeObs = null;
  let lastArticleHeight = -1;

  function isWide() {
    return window.matchMedia(WIDE_QUERY).matches;
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
            "sup span." +
            HOVER_CLASS +
            "{color:#66b3ff;text-shadow:0 0 6px rgba(77,163,255,.9);}";
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
    fn.addEventListener("mouseenter", on);
    fn.addEventListener("mouseleave", off);
    note.addEventListener("mouseenter", on);
    note.addEventListener("mouseleave", off);
    return () => {
      fn.removeEventListener("mouseenter", on);
      fn.removeEventListener("mouseleave", off);
      note.removeEventListener("mouseenter", on);
      note.removeEventListener("mouseleave", off);
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

    if (!isWide() || footnotes.length === 0) {
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

  function init() {
    const article = getArticle();
    if (!article) return;

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
