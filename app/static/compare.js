// app/static/compare.js — compare mode (CHAT_UI_PLAN.md P9-F): one X-ray decoded by two models, drawn as two columns of one card.
//
// Pure, and no DOM: the view state.js folds keeps block 0 where every turn keeps its report and block 1 under view.compare. columnsOf gives
// each block as a view of its own, which render.js draws with the builders of any card (the summary, the sentences, the notes, the label
// chips, Copy and Show raw). Block 1's view is the turn's, with block 1's text, labels and card in place of block 0's, and its own generate
// and label details (each stage's detail.compare): so its note says why its own report ends where it does, and it has no agreement, score or
// retrieval, which stay block 0's. Its id is the turn's with COMPARE_KEY after it, so that its Show raw is a state of its own in ctx.ui.

export const COMPARE_KEY = 'compare';   // block 1's: its ui record (`<message id>:compare`) and the data-key of its Copy and Show raw

const isObject = (v) => v !== null && typeof v === 'object' && !Array.isArray(v);
const str = (v) => (typeof v === 'string' ? v : '');

// A column's header: the model's name, from its card, else from what the turn names it.
export const modelName = (card, fallback) => str(card?.name) || str(fallback) || 'model';

// A stage as block 1 has it: its detail.compare as the detail (none when the stage has no such part); a skipped stage as it is.
function ownStage(stage) {
  if (!isObject(stage) || !isObject(stage.detail)) return stage;
  return { ...stage, detail: isObject(stage.detail.compare) ? stage.detail.compare : {} };
}

function blockOne(v) {
  const c = v.compare;
  const stages = { ...v.stages };
  for (const name of ['generate', 'label']) if (name in stages) stages[name] = ownStage(stages[name]);
  return { ...v, id: `${str(v.id)}:${COMPARE_KEY}`, provenance: isObject(c.card) ? c.card : null, stages, report: str(c.report),
           displayReport: str(c.displayReport), provisional: c.provisional === true, truncated: c.truncated === true,
           labels: isObject(c.labels) ? c.labels : null, agreement: null, score: null, neighbors: [], matches: [], trueRank: null, compare: null };
}

// -> [{ index: 0, model, view }, { index: 1, model, view }] for a compare turn, block 0's view the turn's own; null for any other turn. The columns
// come with the generate stage (P9-F fix 1): a compare turn that ends before it (a question that runs no model, a failure or a Stop before
// generate) decoded with neither model, and draws the card a turn that does not compare draws.
export function columnsOf(view) {
  if (!isObject(view) || !isObject(view.compare) || !isObject(view.stages) || !isObject(view.stages.generate)) return null;
  return [
    { index: 0, model: modelName(view.provenance, view.options?.model), view },
    { index: 1, model: modelName(view.compare.card, view.compare.model), view: blockOne(view) },
  ];
}
