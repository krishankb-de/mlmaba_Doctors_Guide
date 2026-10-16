// app/static/reveal.js: how a turn's answer is shown: held on the reading card while the X-ray is read, then drawn top to bottom.
//
// Two ways to show a turn, a setting of the page (the drawer's "Showing results"), stored apart from the turn's options under cxrchat.display:
// it changes what the page draws and never what the server runs, so it is never sent (app/schemas.py Options refuses any key it does not know).
//   ordered (the default)   From Send the card is the reading card (render.js renderReadingCard): the X-ray on a lightbox and the stages as they
//                           really run. It stays until the turn has ended AND the pause the setting names has passed since Send (5 s unless
//                           changed), so that a fast turn is not thrown at the reader the instant it lands. Then the whole card is drawn and
//                           shown in order: the report line by line, then each part under it (findings, similar X-rays, matching reports, the
//                           rest). A turn that fails or is stopped is shown at once: an error is never held back.
//   live                    The card as it always was: drawn from the first event, the report written while it is decoded.
// The pause is the page's own and is said as such in the drawer; the stages the reading card lists are the turn's real ones, so nothing on it
// claims work that is not being done.
//
// The first half is pure (the stored setting, the phase of a turn at a time, the schedule of the reveal) and node tests it without a page.
// revealCard marks a card that is already built: a class and a delay on each line and part, and data-reveal="run" on the card, which
// styles.css animates (and does not, for a reader who asked for less motion). Nothing here sets a timer or touches the page: app.js does.

export const DISPLAY_KEY = 'cxrchat.display';
export const REVEAL_MODES = Object.freeze(['ordered', 'live']);
export const HOLD_CHOICES = Object.freeze([0, 1, 2, 3, 4, 5]);   // seconds of pause before a finished turn is shown (the drawer's list)
export const DEFAULT_DISPLAY = Object.freeze({ reveal: 'ordered', hold_s: 5 });   // 5 s: real reports on the cluster's CPU server took 3.5-3.7 s (2026-10-16)

const isObject = (v) => v !== null && typeof v === 'object' && !Array.isArray(v);

// Anything in, a complete display setting out: a mode it does not know, or a pause that is not one of the choices, takes the default.
export function sanitizeDisplay(raw) {
  const o = isObject(raw) ? raw : {};
  return {
    reveal: REVEAL_MODES.includes(o.reveal) ? o.reveal : DEFAULT_DISPLAY.reveal,
    hold_s: HOLD_CHOICES.includes(o.hold_s) ? o.hold_s : DEFAULT_DISPLAY.hold_s,
  };
}

// The stored setting, or the default: storage may be absent, may throw on any access, and may hold anything.
export function loadDisplay(storage) {
  try {
    const raw = storage?.getItem(DISPLAY_KEY);
    return sanitizeDisplay(typeof raw === 'string' ? JSON.parse(raw) : null);
  } catch { return sanitizeDisplay(null); }
}

// true when it was stored; false when there is no storage, or it refused (full, blocked, a private window).
export function saveDisplay(storage, display) {
  try {
    storage.setItem(DISPLAY_KEY, JSON.stringify(sanitizeDisplay(display)));
    return true;
  } catch { return false; }
}

// Where a turn's card is at a moment: { phase, waitMs }.
//   reading   the reading card: the turn runs, or it has ended and the pause since Send (sentAt + holdMs) has waitMs left to go
//   reveal    the whole card, shown in order: the turn has ended well and the pause is over (once: `revealed` says it has been)
//   shown     the whole card as it is: a live turn, one shown already, and one that failed or was stopped
// ordered is the turn's own (the setting when it was sent, and only for a turn that runs the model); now and sentAt are milliseconds.
export function revealPhase({ ordered, revealed, status, sentAt, now, holdMs }) {
  if (!ordered || revealed) return { phase: 'shown', waitMs: 0 };
  if (status === 'running') return { phase: 'reading', waitMs: 0 };
  if (status !== 'done') return { phase: 'shown', waitMs: 0 };   // an error or a stop: said at once
  const waitMs = Math.max(0, (Number.isFinite(sentAt) ? sentAt : 0) + (Number.isFinite(holdMs) ? holdMs : 0) - now);
  return waitMs > 0 ? { phase: 'reading', waitMs } : { phase: 'reveal', waitMs: 0 };
}

// The beat of the reveal: a line of the report every lineMs, faster for a long report so that its lines take linesMs at most, then a part
// every partMs; settleMs lets the last one finish before the reveal is over. A report of eight lines and five parts takes about four seconds.
export const REVEAL_TIMING = Object.freeze({ lineMs: 260, linesMs: 2600, partMs: 240, settleMs: 450 });

// kinds: 'line' or 'part' for each unit, in the order the card has them -> { delays (ms, one per unit), totalMs }. Nothing to show takes no time.
export function revealSchedule(kinds) {
  const list = Array.isArray(kinds) ? kinds : [];
  const lines = list.filter((k) => k === 'line').length;
  const step = lines ? Math.min(REVEAL_TIMING.lineMs, REVEAL_TIMING.linesMs / lines) : 0;
  const delays = [];
  let at = 0;
  for (const kind of list) {
    delays.push(Math.round(at));
    at += kind === 'line' ? step : REVEAL_TIMING.partMs;
  }
  return { delays, totalMs: delays.length ? Math.round(delays.at(-1) + REVEAL_TIMING.settleMs) : 0 };
}

// What a card shows in turn. Lines: each sentence of the report (or a paragraph of the plain display) and each section's heading. Parts: what
// comes between and under them: the plain-language summary, the report's notes, its terms and its buttons, every section of the card (findings,
// similar X-rays, matching reports, published lines, images, how it was made), and in a compare card each column's model name and labels.
export const REVEAL_LINES = '.report-section > h3, .report-list > li, .report-section > p';
export const REVEAL_PARTS = '.report-summary, .report > .note, .report-extras, .report-actions, .card > section, .compare-model, .compare-column > .labels';

// Hidden itself, or under something hidden, short of the card: not shown, so it takes no beat.
function hiddenWithin(node, card) {
  for (let n = node; n && n !== card; n = n.parentNode) if (n.hasAttribute?.('hidden')) return true;
  return false;
}

// Marks a built card for its reveal: each line and part in document order gets its class and its delay (--reveal-at), and the card
// data-reveal="run". -> { count, totalMs }: how many units, and how long until the last has finished; totalMs 0 when there is nothing to show
// in turn (the card is then left as it is). finishReveal ends it early: everything shows as it is.
export function revealCard(card) {
  if (typeof card?.querySelectorAll !== 'function') return { count: 0, totalMs: 0 };
  const units = Array.from(card.querySelectorAll(`${REVEAL_LINES}, ${REVEAL_PARTS}`)).filter((n) => !hiddenWithin(n, card));
  const kinds = units.map((n) => (n.matches(REVEAL_LINES) ? 'line' : 'part'));
  const { delays, totalMs } = revealSchedule(kinds);
  if (!totalMs) return { count: 0, totalMs: 0 };
  units.forEach((n, i) => {
    n.classList.add(kinds[i] === 'line' ? 'reveal-line' : 'reveal-part');
    n.style.setProperty('--reveal-at', `${delays[i]}ms`);
  });
  card.setAttribute('data-reveal', 'run');
  return { count: units.length, totalMs };
}

export function finishReveal(card) {
  card?.setAttribute?.('data-reveal', 'done');
}
