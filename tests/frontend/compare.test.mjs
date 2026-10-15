// tests/frontend/compare.test.mjs — compare mode in the page (CHAT_UI_PLAN.md P9-F): state.js folds block 1 under `compare`, compare.js gives each
// block as a view of its own, and render.js draws the two as columns of one card.
//
// Events are the recorded compare turn of the tiny stack (fixtures/turn_compare.json: two tiny engines named after the published models, random
// weights, the keyword labeller) and synthetic turns folded from events written here: no MIMIC data.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { installDom, serialize } from './dom_shim.mjs';
import { applyEvent, initialView, replay } from '../../app/static/state.js';
import * as compareModule from '../../app/static/compare.js';
import * as render from '../../app/static/render.js';
import { renderAssistantCard } from '../../app/static/render.js';

installDom();

const M3 = 'hybrid_150m_m3_rrg';
const D13 = 'hybrid_150m_v2_rrg';
const LABEL_NAMES = [
  'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia',
  'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices', 'No Finding',
];
const recorded = () => JSON.parse(readFileSync(new URL('./fixtures/turn_compare.json', import.meta.url)));
const plainTurn = () => JSON.parse(readFileSync(new URL('./fixtures/turn_tiny.json', import.meta.url)));
const columnsOf = (...args) => {
  assert.equal(typeof compareModule.columnsOf, 'function', 'compare.js exports columnsOf');
  return compareModule.columnsOf(...args);
};

const q = (node, selector) => node.querySelector(selector);
const qa = (node, selector) => node.querySelectorAll(selector);
const texts = (nodes) => nodes.map((n) => n.textContent);
// A sentence of the card as it was written: without what is only drawn on it, a term's number and the ⚠ of a sentence about an earlier study.
const bare = (node) => (node.nodeType === 3 ? node.data
  : node.localName === 'sup' || node.classList.contains('flag') ? '' : node.childNodes.map(bare).join(''));
const column = (card, n) => q(card, `section.compare-column[data-column="${n}"]`);
const deltasOf = (log, index) => log.filter((e) => e.event === 'content_block_delta' && e.data.index === index);
const stopIndex = (log, index) => log.findIndex((e) => e.event === 'content_block_stop' && e.data.index === index);
const stageDetail = (log, stage) => log.find((e) => e.event === 'stage_end' && e.data.stage === stage).data.detail;
const labelsOf = (positives) => Object.fromEntries(LABEL_NAMES.map((n) => [n, positives.includes(n) ? 1 : 0]));

function deepFreeze(value) {
  if (value && typeof value === 'object' && !Object.isFrozen(value)) {
    Object.freeze(value);
    Object.values(value).forEach(deepFreeze);
  }
  return value;
}

// A synthetic compare turn: block 0 and block 1 each with its report, its labels and its generate detail, as the server streams one.
function compareLog({ report0 = 'Findings: The lungs are clear. Impression: No acute disease.', report1 = 'Findings: Mild cardiomegaly. Impression: Stable.',
                      labels0 = labelsOf(['No Finding']), labels1 = labelsOf(['Cardiomegaly']), generate0 = {}, generate1 = {}, truncated1 = false,
                      display1 = null, options = {} } = {}) {
  const steps = [
    ['message_start', { message_id: 'm_c', user_message_id: 'u_c', session_id: 's_c', mode: 'private',
                        model: { name: M3, device: 'cpu', prefix_k: 32 },
                        options: { model: M3, decode: 'beam', beam_size: 3, max_new_tokens: 100, cached_decode: false, compile: false, k_images: 4,
                                   k_reports: 3, label: true, reference: null, display_repair: false, stop_on_repeat: false, test_row: null, compare: true, ...options },
                        image: { sha256: 'ab'.repeat(32), filename: 'chest.png', source: 'upload', urls: {} },
                        compare: { model: D13, card: { name: D13, device: 'cpu', prefix_k: 32 } } }],
    ['stage_start', { stage: 'preprocess', index: 0 }], ['stage_end', { stage: 'preprocess', ms: 2, detail: { input_px: [320, 320] } }],
    ['stage_start', { stage: 'encode', index: 1 }], ['stage_end', { stage: 'encode', ms: 5, detail: { prefix_tokens: 32, compare: { prefix_tokens: 32 } } }],
    ['stage_end', { stage: 'retrieve', skipped: 'gallery_unavailable' }],
    ['stage_start', { stage: 'generate', index: 3 }],
    ['content_block_start', { index: 0, content_block: { type: 'report', text: '' } }],
    ['content_block_delta', { index: 0, delta: { type: 'beam_snapshot', step: 0, text: report0 } }],
    ['content_block_stop', { index: 0 }],
    ['content_block_start', { index: 1, content_block: { type: 'report', text: '' } }],
    ['content_block_delta', { index: 1, delta: { type: 'beam_snapshot', step: 0, text: report1 } }],
    ['content_block_stop', { index: 1 }],
    ['stage_end', { stage: 'generate', ms: 9, detail: { decode: 'beam', tokens: 87, stopped: 'budget', ...generate0,
                                                        compare: { decode: 'beam', tokens: 64, stopped: 'budget', ...generate1 } } }],
    ['stage_start', { stage: 'label', index: 4 }],
    ['stage_end', { stage: 'label', ms: 3, detail: { chexbert_14: labels0, positives: [], neighbor_agreement: [],
                                                     compare: { chexbert_14: labels1, positives: [] } } }],
    ['stage_end', { stage: 'score', skipped: 'no_reference' }],
    ['message_stop', { message_id: 'm_c', status: 'done', total_ms: 30, report: report0, display_report: report0, truncated_mid_sentence: false,
                       disclaimer: 'Research prototype; not for clinical use.',
                       compare: { model: D13, report: report1, display_report: display1 ?? report1, truncated_mid_sentence: truncated1 } }],
  ];
  return steps.map(([event, data], i) => ({ event, data: { ...data, seq: i + 1 } }));
}

// ---- the reducer -----------------------------------------------------------------------------------------------------------------

test('the reducer folds block 0 where every turn keeps its report and block 1 under compare, from the recorded compare turn', () => {
  const log = recorded();
  const stop = log.at(-1).data;
  const view = replay(log);
  assert.deepEqual([view.status, view.report, view.displayReport, view.provisional], ['done', stop.report, stop.display_report, false]);
  assert.deepEqual(view.compare, {
    model: D13, card: log[0].data.compare.card, started: true, report: stop.compare.report, displayReport: stop.compare.display_report,
    provisional: false, truncated: stop.compare.truncated_mid_sentence, labels: stageDetail(log, 'label').compare.chexbert_14,
  });
  assert.deepEqual(view.labels, stageDetail(log, 'label').chexbert_14);   // block 0's own, as for every turn
  assert.notEqual(view.compare.report, view.report);
});

test('live, replayed and polled folds of a compare turn are the same view, and the reducer never writes to what it is given', () => {
  const log = recorded();
  let live = initialView(log[0].data.message_id);
  for (const e of log) live = applyEvent(deepFreeze(live), deepFreeze(structuredClone(e)));
  assert.deepEqual(live, replay(log));
  const polled = [...log.slice(0, 30), ...log.slice(10, 40), ...log.slice(25)];   // a poll resends what the stream delivered
  assert.deepEqual(replay(polled), replay(log));
});

test('block 0 streams, ends, and only then block 1 streams: each block keeps its own text and its own provisional flag', () => {
  const log = recorded();
  const firstOf = (index, n) => log.indexOf(deltasOf(log, index)[n - 1]);
  const at = (i) => replay(log.slice(0, i + 1));
  const block0 = at(firstOf(0, 3));
  assert.deepEqual([block0.provisional, block0.report, block0.compare.report, block0.compare.provisional],
                   [true, deltasOf(log, 0)[2].data.delta.text, '', false]);   // block 1 has not started
  const between = at(stopIndex(log, 0));
  assert.deepEqual([between.provisional, between.compare.provisional], [false, false]);
  const block1 = at(firstOf(1, 2));
  assert.deepEqual([block1.provisional, block1.report, block1.compare.provisional, block1.compare.report],
                   [false, deltasOf(log, 0).at(-1).data.delta.text, true, deltasOf(log, 1)[1].data.delta.text]);
  assert.equal(at(stopIndex(log, 1)).compare.provisional, false);
});

test('an aborted compare turn keeps what each block streamed, and a turn that never compares has no compare at all', () => {
  const log = recorded();
  const cut = log.slice(0, log.indexOf(deltasOf(log, 1)[2]) + 1);
  const aborted = replay([...cut, { event: 'message_stop', data: { message_id: log[0].data.message_id, status: 'aborted', total_ms: 9, report: null,
                                                                    display_report: null, truncated_mid_sentence: false, seq: cut.length + 1,
                                                                    compare: { model: D13, report: null, display_report: null, truncated_mid_sentence: false } } }]);
  assert.deepEqual([aborted.status, aborted.provisional, aborted.report], ['aborted', false, deltasOf(log, 0).at(-1).data.delta.text]);
  assert.deepEqual([aborted.compare.provisional, aborted.compare.report, aborted.compare.displayReport],
                   [false, deltasOf(log, 1)[2].data.delta.text, deltasOf(log, 1)[2].data.delta.text]);
  assert.equal(replay(plainTurn()).compare, null);
  assert.equal(initialView('m').compare, null);
});

// ---- compare.js: each block as a view of its own ---------------------------------------------------------------------------------

test('columnsOf gives no columns for a turn that does not compare, and each block as a view of its own for one that does', () => {
  assert.equal(columnsOf(replay(plainTurn())), null);
  assert.equal(columnsOf(initialView('m')), null);
  const log = compareLog({ generate1: { stopped: 'repeat' } });
  const view = replay(log);
  const [zero, one] = columnsOf(view);
  assert.deepEqual([zero.index, zero.model, one.index, one.model], [0, M3, 1, D13]);
  assert.equal(zero.view, view);   // block 0 is the turn's own view
  const b = one.view;
  assert.deepEqual([b.report, b.displayReport, b.labels, b.provenance], [view.compare.report, view.compare.displayReport, view.compare.labels, view.compare.card]);
  assert.deepEqual([b.score, b.agreement, b.neighbors, b.matches, b.trueRank, b.compare], [null, null, [], [], null, null]);
  assert.equal(b.stages.generate.detail.stopped, 'repeat');                       // its own stop, for its own note
  assert.deepEqual(b.stages.label.detail, stageDetail(log, 'label').compare);     // its own labels, and no agreement of block 0's
  assert.deepEqual([b.status, b.options, b.mode, b.id], [view.status, view.options, view.mode, `${view.id}:compare`]);   // its ui record is its own
  assert.equal(view.stages.generate.detail.stopped, 'budget');                   // the turn's view is not touched
});

// ---- the card ----------------------------------------------------------------------------------------------------------------------

test('a compare card has two columns, each with its model, summary, sentences, notes, chips, Copy and Show raw; the rest of the card once', () => {
  const copied = [];
  const card = renderAssistantCard(replay(compareLog()), { copy: (t) => { copied.push(t); }, labelNames: LABEL_NAMES, turn: 2, ui: new Map() });
  assert.deepEqual(card.getAttribute('class').split(' '), ['card', 'compare-card']);
  const columns = qa(card, 'section.compare-column');
  assert.deepEqual(columns.map((c) => c.getAttribute('data-column')), ['0', '1']);
  assert.deepEqual(columns.map((c) => q(c, 'h3.compare-model').textContent), [M3, D13]);
  assert.deepEqual(columns.map((c) => c.getAttribute('aria-label')), [`Report of ${M3}, turn 2`, `Report of ${D13}, turn 2`]);
  assert.deepEqual(columns.map((c) => q(c, '.summary-box p').textContent),
                   ['The report mentions no abnormal finding.', 'The report mentions: an enlarged heart (cardiomegaly).']);
  assert.deepEqual(columns.map((c) => qa(c, '.report-body li').map(bare)), [['The lungs are clear.', 'No acute disease.'], ['Mild cardiomegaly.', 'Stable.']]);
  assert.deepEqual(columns.map((c) => qa(c, '.label-chips li').length), [14, 14]);
  assert.deepEqual(columns.map((c) => texts(qa(c, '.label-chips li.positive')).map((t) => t.split(':')[0])), [['No Finding'], ['Cardiomegaly']]);
  assert.deepEqual(columns.map((c) => q(c, '.label-chips').getAttribute('aria-label')),
                   [`CheXbert-14 labels of the ${M3} report`, `CheXbert-14 labels of the ${D13} report`]);
  assert.deepEqual(columns.map((c) => qa(c, 'button').map((b) => [b.getAttribute('data-action'), b.getAttribute('data-key'), b.getAttribute('aria-label')])), [
    [['copy', null, `Copy report of ${M3}, turn 2`], ['raw', null, `Show raw report of ${M3}, turn 2`]],
    [['copy', 'compare', `Copy report of ${D13}, turn 2`], ['raw', 'compare', `Show raw report of ${D13}, turn 2`]]]);
  // once, outside the columns: the timeline, the notes, the provenance; and nothing of a report outside them
  assert.equal(qa(card, 'ol.timeline').length, 1);
  assert.equal(qa(card, 'footer.provenance').length, 1);
  assert.equal(qa(card, '.report').length, 2);
  assert.ok(qa(card, '.report').every((r) => r.closest('section.compare-column')));
  assert.equal(q(card, '.compare-waiting'), null);
});

test('Copy and Show raw work per column: each copies its own text, and Show raw changes its own column and nothing else', () => {
  const copied = [];
  const ui = new Map();
  const view = replay(compareLog({ report1: 'Findings: Mild cardiomegaly. Impression: Stable. Impression: Stable.', display1: 'Findings: Mild cardiomegaly. Impression: Stable.' }));
  const ctx = { copy: (t) => { copied.push(t); }, labelNames: LABEL_NAMES, turn: 1, ui };
  let card = renderAssistantCard(view, ctx);
  const button = (n, action) => q(column(card, n), `button[data-action="${action}"]`);
  button(1, 'copy').click();
  button(0, 'copy').click();
  assert.deepEqual(copied, [view.compare.displayReport, view.displayReport]);   // the clean text of each column
  button(1, 'raw').click();
  assert.equal(q(column(card, 1), 'pre.report-raw').textContent, view.compare.report);
  assert.equal(q(column(card, 0), 'pre.report-raw'), null);
  assert.ok(q(column(card, 0), '.report-body li'));
  button(1, 'copy').click();
  assert.equal(copied.at(-1), view.compare.report);                            // raw on: Copy hands over the raw text, as on any card
  assert.deepEqual([ui.get('m_c').raw, ui.get('m_c:compare').raw], [false, true]);
  card = renderAssistantCard(view, ctx);                                        // a rebuilt card keeps each column's own state
  assert.equal(q(column(card, 1), 'pre.report-raw').textContent, view.compare.report);
  assert.equal(q(column(card, 0), 'pre.report-raw'), null);
});

test('each column is a report card of its own: its summary from its labels, its numbering of terms, its notes', () => {
  const report0 = 'Findings: Small pleural effusion. Compared with ___, it is stable. Impression: Effusion.';
  const report1 = 'Findings: Basilar atelectasis and a nodule. Impression: Atelectasis.';
  const card = renderAssistantCard(replay(compareLog({ report0, report1, labels0: labelsOf(['Pleural Effusion']), labels1: labelsOf(['Atelectasis', 'Lung Lesion']) })),
                                   { labelNames: LABEL_NAMES, ui: new Map() });
  const terms = (n) => qa(column(card, n), '.term').map((t) => [q(t, 'sup').textContent, t.textContent.replace(/\d+$/, '')]);
  assert.deepEqual(terms(0), [['1', 'pleural effusion']]);
  assert.deepEqual(terms(1), [['1', 'Basilar'], ['2', 'atelectasis'], ['3', 'nodule']]);      // numbered from 1 in each column
  assert.deepEqual(qa(column(card, 0), '.report-key').map((p) => p.getAttribute('data-note')), ['removed', 'earlier']);
  assert.deepEqual(qa(column(card, 1), '.report-key').map((p) => p.getAttribute('data-note')), []);
  assert.equal(qa(column(card, 0), '.removed').length, 1);
  assert.equal(qa(column(card, 0), 'li.earlier .flag').length, 1);
  assert.deepEqual(qa(column(card, 1), '.term-list li').length, 3);
  assert.equal(q(column(card, 1), '.summary-box p').textContent,
               'The report mentions: a spot or mass in the lung (lung lesion); part of a lung is partly collapsed or not fully inflated (atelectasis).');
});

test('each column says why its own report ends where it does', () => {
  const card = renderAssistantCard(replay(compareLog({ generate0: { stopped: 'budget' }, generate1: { stopped: 'repeat' } })), { ui: new Map() });
  assert.deepEqual(texts(qa(column(card, 0), '.report .note')), []);
  assert.deepEqual(texts(qa(column(card, 1), '.report .note')), ['Stopped when the model began repeating itself.']);
  const cut = renderAssistantCard(replay(compareLog({ truncated1: true })), { ui: new Map() });
  assert.deepEqual(texts(qa(column(cut, 1), '.report .note')), ['Report stopped at the token budget mid-sentence']);
  assert.deepEqual(texts(qa(column(cut, 0), '.report .note')), []);
});

test('a column that is still being written keeps the plain display; column 1 says it starts once column 0 is done', () => {
  const log = recorded();
  const at = (i) => renderAssistantCard(replay(log.slice(0, i + 1)), { copy() {}, labelNames: LABEL_NAMES, ui: new Map() });
  const during0 = at(log.indexOf(deltasOf(log, 0)[4]));
  assert.ok(q(column(during0, 0), '.report.provisional') && q(column(during0, 0), '.report-section p'));
  assert.equal(q(column(during0, 0), '.report-body li'), null);
  assert.equal(q(column(during0, 0), 'button[data-action="copy"]'), null);    // no Copy while the text can still change
  assert.ok(q(column(during0, 1), '.report[hidden]'));
  assert.equal(q(column(during0, 1), '.compare-waiting').textContent, `Starts once the ${M3} report is done.`);
  const during1 = at(log.indexOf(deltasOf(log, 1)[4]));
  assert.ok(q(column(during1, 0), '.report-body li') && !q(column(during1, 0), '.report.provisional'));
  assert.ok(q(column(during1, 1), '.report.provisional') && q(column(during1, 1), '.report-section p'));
  assert.equal(q(column(during1, 1), '.compare-waiting'), null);
  assert.equal(q(column(during1, 1), 'button[data-action="copy"]'), null);
});

test('the focus keys of a column\'s Copy and Show raw name their column, so a rebuilt card gives the focus back to the same one', () => {
  const ctx = { copy() {}, labelNames: LABEL_NAMES, ui: new Map() };
  const view = replay(compareLog());
  const first = renderAssistantCard(view, ctx);
  document.body.replaceChildren(first);
  const raw1 = q(column(first, 1), 'button[data-action="raw"]');
  assert.deepEqual([render.focusKey(raw1), render.focusKey(q(column(first, 0), 'button[data-action="raw"]'))], ['raw:compare', 'raw']);
  assert.equal(render.focusKey(q(column(first, 1), 'button[data-action="copy"]')), 'copy:compare');
  const next = renderAssistantCard(view, ctx);
  first.replaceWith(next);
  assert.equal(render.restoreFocus(next, 'raw:compare'), true);
  assert.equal(document.activeElement, q(column(next, 1), 'button[data-action="raw"]'));
  assert.equal(render.restoreFocus(next, 'copy'), true);
  assert.equal(document.activeElement, q(column(next, 0), 'button[data-action="copy"]'));
});

// ---- P9-F fix 1: no columns for a turn whose decodes never started ------------------------------------------------------------------

// A turn's log, as the server streams it, without what compare adds to it: what a turn that does not compare would have streamed.
const withoutCompare = (log) => log.map((e) => {
  const { compare: _dropped, ...data } = e.data;
  return { event: e.event, data: e.event === 'message_start' ? { ...data, options: Object.fromEntries(Object.entries(data.options).filter(([k]) => k !== 'compare')) } : data };
});
const QUESTION = 'I generate chest X-ray reports, I do not answer questions.';

test('a question turn sent with compare on draws the card a question draws, whatever the log says of compare', () => {
  const start = compareLog()[0].data;
  const stop = compareLog().at(-1).data;
  const asked = (withCompareFields) => [
    { event: 'message_start', data: { ...start, image: null, seq: 1, ...(withCompareFields ? {} : { compare: undefined }) } },
    { event: 'warning', data: { code: 'not_a_command', message: QUESTION, seq: 2 } },
    { event: 'message_stop', data: { ...stop, report: null, display_report: null, seq: 3,
                                     compare: withCompareFields ? { model: D13, report: null, display_report: null, truncated_mid_sentence: false } : undefined } },
  ].map((e) => ({ event: e.event, data: JSON.parse(JSON.stringify(e.data)) }));   // undefined keys gone, as on the wire
  const ctx = () => ({ copy() {}, labelNames: LABEL_NAMES, turn: 1, ui: new Map() });
  const plain = serialize(renderAssistantCard(replay(withoutCompare(asked(false))), ctx()));
  assert.equal(serialize(renderAssistantCard(replay(asked(false)), ctx())), plain);   // as the server streams it now: no compare fields at all
  assert.equal(serialize(renderAssistantCard(replay(asked(true)), ctx())), plain);    // and a log that has them (the first server of P9-F) draws no columns
  assert.equal(columnsOf(replay(asked(true))), null);
});

test('a compare turn that ends before its generate stage starts draws exactly the card a turn that does not compare draws', () => {
  const log = compareLog();
  const cut = [log[0], log[1],   // message_start, preprocess started; then the turn fails there, or is stopped
               { event: 'error', data: { type: 'error', error: { type: 'model_error', message: 'Internal error (RuntimeError)' }, seq: 3 } },
               { event: 'message_stop', data: { ...log.at(-1).data, status: 'error', report: null, display_report: null, seq: 4,
                                                compare: { model: D13, report: null, display_report: null, truncated_mid_sentence: false } } }];
  const ctx = () => ({ copy() {}, labelNames: LABEL_NAMES, turn: 1, ui: new Map() });
  for (const ending of [cut, [...cut.slice(0, 2), { ...cut[3], data: { ...cut[3].data, status: 'aborted', seq: 3 } }]]) {
    const card = renderAssistantCard(replay(ending), ctx());
    assert.equal(card.getAttribute('class'), 'card');
    assert.equal(q(card, '.compare-columns'), null);
    assert.equal(serialize(card), serialize(renderAssistantCard(replay(withoutCompare(ending)), ctx())));
  }
  const started = replay([...log.slice(0, 6), { event: 'stage_start', data: { stage: 'generate', index: 3, seq: 7 } }]);
  assert.notEqual(columnsOf(started), null);   // the columns come with the generate stage, before either block has a word
});

// ---- P9-F fix 1: an error in block 1, the waiting note, the Copy that fails -------------------------------------------------------------

const tick = () => new Promise((resolve) => setImmediate(resolve));
const ENDED_IN_ERROR = (seq) => [
  { event: 'error', data: { type: 'error', error: { type: 'model_error', message: 'Internal error (RuntimeError)' }, seq } },
  { event: 'message_stop', data: { message_id: 'm_c', status: 'error', total_ms: 30, report: null, display_report: null, truncated_mid_sentence: false,
                                   disclaimer: 'Research prototype; not for clinical use.', seq: seq + 1,
                                   compare: { model: D13, report: null, display_report: null, truncated_mid_sentence: false } } },
];

test('an error in block 1 leaves column 1 the text it had streamed, as block 0 keeps its own text on an error, and the card says the error once', () => {
  const log = compareLog();
  const firstSnapshot = log.findIndex((e) => e.event === 'content_block_delta' && e.data.index === 1);
  const failing = [...log.slice(0, firstSnapshot + 1), ...ENDED_IN_ERROR(firstSnapshot + 2)];
  const view = replay(failing);
  assert.deepEqual([view.status, view.compare.report, view.compare.provisional], ['error', log[firstSnapshot].data.delta.text, false]);
  const card = renderAssistantCard(view, { labelNames: LABEL_NAMES, ui: new Map() });
  assert.deepEqual(qa(column(card, 1), '.report-body li').map(bare), ['Mild cardiomegaly.', 'Stable.']);   // what streamed, kept
  assert.deepEqual(qa(column(card, 0), '.report-body li').map(bare), ['The lungs are clear.', 'No acute disease.']);
  assert.deepEqual(texts(qa(card, '.notes .note.error')), ['Error: Internal error (RuntimeError)']);           // once, for the card
});

test('the waiting note goes when block 1 starts, and until its first snapshot column 1 shows what column 0 shows before its own', () => {
  const log = recorded();
  const at = (i) => renderAssistantCard(replay(log.slice(0, i + 1)), { copy() {}, labelNames: LABEL_NAMES, ui: new Map() });
  const start = (index) => log.findIndex((e) => e.event === 'content_block_start' && e.data.index === index);
  const ended0 = stopIndex(log, 0);
  assert.equal(replay(log.slice(0, ended0 + 1)).compare.started, false);
  assert.equal(q(column(at(ended0), 1), '.compare-waiting').textContent, `Starts once the ${M3} report is done.`);   // block 0 done, block 1 not begun
  assert.equal(replay(log.slice(0, start(1) + 1)).compare.started, true);   // state.js records block 1's start
  const begun = column(at(start(1)), 1);
  assert.equal(q(begun, '.compare-waiting'), null);                          // it has begun: the note is no longer true
  const before0 = column(at(start(0)), 0);                                   // block 0 begun, and no snapshot of it yet
  assert.equal(serialize(q(begun, '.report')), serialize(q(before0, '.report')));
  assert.equal(serialize(q(begun, '.labels')), serialize(q(before0, '.labels')));
});

test('a column\'s Copy that fails says so with the model\'s name, as its idle label does; a card\'s one Copy says it as before', async (t) => {
  const timers = [];
  const realSetTimeout = globalThis.setTimeout;
  globalThis.setTimeout = (fn, ms) => { timers.push({ fn, ms }); return timers.length; };   // a timer is run by hand
  t.after(() => { globalThis.setTimeout = realSetTimeout; });
  const card = renderAssistantCard(replay(compareLog()), { copy: () => Promise.reject(new Error('denied')), labelNames: LABEL_NAMES, turn: 2, ui: new Map() });
  for (const [n, model] of [[1, D13], [0, M3]]) {
    timers.length = 0;
    const copy = q(column(card, n), 'button[data-action="copy"]');
    copy.click();
    await tick();
    assert.deepEqual([copy.textContent, copy.getAttribute('aria-label')], ['Copy failed', `Copy failed for the report of ${model}, turn 2`]);
    timers[0].fn();   // the moment is over
    assert.deepEqual([copy.textContent, copy.getAttribute('aria-label')], ['Copy', `Copy report of ${model}, turn 2`]);
  }
  const single = renderAssistantCard(replay(plainTurn()), { copy: () => Promise.reject(new Error('denied')), turn: 2, ui: new Map() });
  q(single, 'button[data-action="copy"]').click();
  await tick();
  assert.equal(q(single, 'button[data-action="copy"]').getAttribute('aria-label'), 'Copy failed, turn 2');   // as it always was
});

// ---- P9-F fix 2: a tiny engine under a published model's name is marked a test model ---------------------------------------------------

test('a card that says test_model is a test model, whatever it is called, and its turn\'s card carries the test notice', () => {
  assert.equal(render.isTestModel({ name: M3, test_model: true }), true);
  for (const other of [{ name: M3 }, { name: M3, test_model: false }, { name: M3, test_model: 'true' }, { name: M3, test_model: 1 }]) {
    assert.equal(render.isTestModel(other), false, JSON.stringify(other));   // only a real true says it
  }
  const notice = (card) => q(card, 'p.test-notice')?.textContent ?? null;
  const fixture = recorded();
  assert.equal(fixture[0].data.model.test_model, true);                  // the recorded tiny compare turn: both cards renamed, both marked
  assert.equal(notice(renderAssistantCard(replay(fixture), { ui: new Map() })), render.TEST_NOTICE);
  const second = compareLog().map((e, i) => (i === 0 ? { ...e, data: { ...e.data, compare: { model: D13, card: { name: D13, test_model: true } } } } : e));
  assert.equal(notice(renderAssistantCard(replay(second), { ui: new Map() })), render.TEST_NOTICE);   // either column's model makes it one
  assert.equal(notice(renderAssistantCard(replay(compareLog()), { ui: new Map() })), null);             // two real cards: no notice
});

test('a card that does not compare is the card it always was', () => {
  const ctx = () => ({ copy() {}, labelNames: LABEL_NAMES, turn: 1, ui: new Map() });
  const card = renderAssistantCard(replay(plainTurn()), ctx());
  assert.equal(card.getAttribute('class'), 'card');
  assert.equal(qa(card, '*').filter((n) => (n.getAttribute('class') ?? '').includes('compare')).length, 0);
  assert.equal(qa(card, '[data-key]').filter((n) => ['copy', 'raw'].includes(n.getAttribute('data-action'))).length, 0);
  assert.deepEqual(qa(card, '.report button').map((b) => b.getAttribute('aria-label')), ['Copy report, turn 1', 'Show raw report, turn 1']);
  assert.equal(q(card, '.label-chips'), null);   // the tiny turn has no labeller: no chips, as before
  assert.deepEqual(card.children.map((c) => c.localName + (c.getAttribute('class') ? `.${c.getAttribute('class')}` : '')),
                   ['ol.timeline', 'div.notes', 'p.test-notice', 'div.report', 'div.labels', 'footer.provenance']);   // its parts, in their order
  assert.ok(serialize(card).includes('Copy report, turn 1'));
});
