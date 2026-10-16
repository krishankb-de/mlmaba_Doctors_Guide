// tests/frontend/reveal.test.mjs: app/static/reveal.js: the display setting, the phase of a turn at a moment, and the schedule of the
// ordered reveal. Pure functions, no page; revealCard on a built card is in render.test.mjs, where the cards are.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  DEFAULT_DISPLAY, DISPLAY_KEY, HOLD_CHOICES, REVEAL_TIMING, loadDisplay, revealPhase, revealSchedule, sanitizeDisplay, saveDisplay,
} from '../../app/static/reveal.js';

const memoryStorage = (initial = {}) => {
  const data = new Map(Object.entries(initial));
  return { data, getItem: (k) => (data.has(k) ? data.get(k) : null), setItem: (k, v) => { data.set(k, String(v)); } };
};
const brokenStorage = () => ({ getItem() { throw new Error('blocked'); }, setItem() { throw new Error('blocked'); } });

test('the display setting: ordered with a 5 s pause unless set, anything else is the default, and it is kept under cxrchat.display', () => {
  assert.deepEqual(DEFAULT_DISPLAY, { reveal: 'ordered', hold_s: 5 });
  assert.equal(DISPLAY_KEY, 'cxrchat.display');
  assert.deepEqual(HOLD_CHOICES, [0, 1, 2, 3, 4, 5]);
  for (const raw of [undefined, null, 42, 'live', [], { reveal: 'fast', hold_s: 7 }, { hold_s: '2' }, { hold_s: 2.5 }]) {
    const s = sanitizeDisplay(raw);
    assert.equal(s.reveal, 'ordered', JSON.stringify(raw));
    assert.equal(s.hold_s, 5, JSON.stringify(raw));
  }
  assert.deepEqual(sanitizeDisplay({ reveal: 'live', hold_s: 0, other: 'x' }), { reveal: 'live', hold_s: 0 });   // a key it does not know is dropped
  for (const storage of [undefined, null, memoryStorage(), brokenStorage(), memoryStorage({ [DISPLAY_KEY]: '{not json' })]) {
    assert.deepEqual(loadDisplay(storage), DEFAULT_DISPLAY);
  }
  const storage = memoryStorage();
  assert.equal(saveDisplay(storage, { reveal: 'live', hold_s: 5 }), true);
  assert.deepEqual(JSON.parse(storage.data.get(DISPLAY_KEY)), { reveal: 'live', hold_s: 5 });
  assert.deepEqual(loadDisplay(storage), { reveal: 'live', hold_s: 5 });
  assert.equal(saveDisplay(brokenStorage(), DEFAULT_DISPLAY), false);   // a storage that refuses says so, and throws nothing
  assert.equal(saveDisplay(undefined, DEFAULT_DISPLAY), false);
});

test('the phase of a turn: reading while it runs and through the pause after it ends, then the reveal once; never held when live, failed or stopped', () => {
  const at = (over) => revealPhase({ ordered: true, revealed: false, status: 'running', sentAt: 1000, now: 1500, holdMs: 3000, ...over });
  assert.deepEqual(at({}), { phase: 'reading', waitMs: 0 });                          // the turn runs
  assert.deepEqual(at({ status: 'done' }), { phase: 'reading', waitMs: 2500 });       // done 0.5 s after Send: 2.5 s of the pause to go
  assert.deepEqual(at({ status: 'done', now: 3999 }), { phase: 'reading', waitMs: 1 });
  assert.deepEqual(at({ status: 'done', now: 4000 }), { phase: 'reveal', waitMs: 0 }); // the pause is over: shown in order
  assert.deepEqual(at({ status: 'done', now: 9000 }), { phase: 'reveal', waitMs: 0 }); // a turn slower than the pause is shown as it ends
  assert.deepEqual(at({ status: 'done', now: 9000, revealed: true }), { phase: 'shown', waitMs: 0 });   // once: a later draw is the card as it is
  assert.deepEqual(at({ status: 'done', holdMs: 0 }), { phase: 'reveal', waitMs: 0 });   // no pause
  for (const status of ['error', 'aborted']) assert.deepEqual(at({ status }), { phase: 'shown', waitMs: 0 }, status);   // never held back
  for (const status of ['running', 'done', 'error']) assert.deepEqual(at({ ordered: false, status }), { phase: 'shown', waitMs: 0 }, status);
});

test('the reveal schedule: a line every 260 ms, faster for a long report, then a part every 240 ms; nothing to show takes no time', () => {
  assert.deepEqual(revealSchedule([]), { delays: [], totalMs: 0 });
  assert.deepEqual(revealSchedule(['part', 'line', 'line', 'line', 'part', 'part']),
                   { delays: [0, 240, 500, 760, 1020, 1260], totalMs: 1260 + REVEAL_TIMING.settleMs });
  const long = revealSchedule([...Array(20).fill('line'), 'part']);   // twenty lines share 2.6 s
  assert.equal(long.delays[1], 130);
  assert.equal(long.delays[20], 2600);
  assert.ok(long.totalMs <= REVEAL_TIMING.linesMs + REVEAL_TIMING.settleMs);
  assert.deepEqual(revealSchedule(['part', 'part']).delays, [0, 240]);   // parts alone keep their beat
});
