// app/static/state.js — pure reducers: the stored event log rebuilds exactly the live view.
//
// applyEvent folds one {event, data} (from the stream or from polling) into a new view and never mutates what it is
// given; replay folds a stored log, so a reload shows what was streamed. Stage keys are inserted in arrival order, so
// Object.keys(view.stages) follows the contract order. view.stages is the log's record and nothing more: a turn that
// ends aborted or in error never sends the stage_end of the stage it was in, nor any event for a stage it never
// reached, so that record still says 'running' or says nothing. stageState(view, name) is what to draw: it closes those.
// The fields a stage fills (neighbors, labels, score, ...) stay empty (null or []) when the stage was skipped.
// STAGES is frozen: every card draws from it, so a renderer that sorts or pushes to it throws instead of reordering them all.
// A compare turn (P9-F) streams a second report, content block 1, by a second model: view.compare holds it (its model and card from
// message_start, whether block 1 has started, its text as it streams and as it ended, its labels from the label stage's detail.compare).
// Block 0 is where every turn keeps its report. compare is null for a turn that does not compare, and every event of such a turn folds as it
// always did.
export const STAGES = Object.freeze(['preprocess', 'encode', 'retrieve', 'generate', 'label', 'score']);   // the contract order

export function initialView(messageId) {
  return { id: messageId, status: 'running', lastSeq: 0, mode: null, provenance: null, options: null,
           image: null, stages: {}, report: '', displayReport: '', provisional: false, truncated: false,
           neighbors: [], matches: [], trueRank: null, labels: null, agreement: null, score: null,
           notices: [], error: null, totalMs: null, compare: null };
}

const isObject = (x) => x !== null && typeof x === 'object' && !Array.isArray(x);
const freshCompare = () => ({ model: null, card: null, started: false, report: '', displayReport: '', provisional: false, truncated: false,
                              labels: null });
// The view with its compare block changed by patch: a block 1 event that comes before message_start (a log cut short) starts one.
const withCompare = (v, patch) => ({ ...v, compare: { ...(v.compare ?? freshCompare()), ...patch } });

export function applyEvent(view, { event, data }) {
  if (data.seq <= view.lastSeq) return view;            // polling can resend what the stream delivered
  const v = { ...view, lastSeq: data.seq };
  switch (event) {
    case 'message_start': {
      const next = { ...v, id: data.message_id, mode: data.mode, provenance: data.model, options: data.options, image: data.image };
      return isObject(data.compare) ? withCompare(next, { model: data.compare.model ?? null, card: data.compare.card ?? null }) : next;
    }
    case 'stage_start':
      return { ...v, stages: { ...v.stages, [data.stage]: { state: 'running' } } };
    case 'stage_end': {
      const st = data.skipped ? { state: 'skipped', skipped: data.skipped }
                              : { state: 'done', ms: data.ms, detail: data.detail };
      const next = { ...v, stages: { ...v.stages, [data.stage]: st } };
      const d = data.skipped ? {} : data.detail || {};         // a skipped stage fills none of the derived fields
      if (data.stage === 'retrieve') return { ...next, neighbors: d.image_neighbors || [], matches: d.report_matches || [], trueRank: d.true_report_rank || null };
      if (data.stage === 'label') {
        const labelled = { ...next, labels: d.chexbert_14 || null, agreement: d.neighbor_agreement || null };
        return v.compare ? withCompare(labelled, { labels: d.compare?.chexbert_14 || null }) : labelled;   // block 1's, under compare
      }
      if (data.stage === 'score') return { ...next, score: data.skipped ? null : d };
      return next;
    }
    case 'content_block_start':   // block 0's start changes nothing a card draws; block 1's ends its column's wait (P9-F fix 1)
      return data.index === 1 ? withCompare(v, { started: true }) : v;
    case 'content_block_delta':
      if (data.index === 1) return withCompare(v, { started: true, report: data.delta.text, provisional: true });
      return { ...v, report: data.delta.text, provisional: true };
    case 'content_block_stop':
      if (data.index === 1) return withCompare(v, { provisional: false });
      return { ...v, provisional: false };
    case 'warning':
      return { ...v, notices: [...v.notices, data] };
    case 'error':
      return { ...v, error: data.error };
    case 'message_stop': {
      const next = { ...v, status: data.status, report: data.report ?? v.report, displayReport: data.display_report ?? v.report,
                     truncated: !!data.truncated_mid_sentence, provisional: false, totalMs: data.total_ms };
      if (!v.compare && !isObject(data.compare)) return next;
      const c = v.compare ?? freshCompare();   // block 1 ends by block 0's rule: what message_stop says, else what streamed
      const e = isObject(data.compare) ? data.compare : {};
      return withCompare(next, { model: c.model ?? e.model ?? null, report: e.report ?? c.report, displayReport: e.display_report ?? c.report,
                                 truncated: !!e.truncated_mid_sentence, provisional: false });
    }
    default:
      return v;
  }
}

export const replay = (events) => events.reduce(applyEvent, initialView(events.length ? events[0].data.message_id : null));

// What the timeline draws for a stage: the log's record while the turn runs, and a settled one once it has ended: a
// stage that was running or never reached is not left spinning or pending.
//   aborted: running and pending -> {state: 'skipped', skipped: 'stopped'}
//   error:   running -> {state: 'error'}, pending -> {state: 'skipped', skipped: 'not_run'}
//   done:    pending -> {state: 'skipped', skipped: 'not_run'}, once the log holds some other stage. Public mode never
//            sends the score stage, so a finished public turn has a stage its log never mentions. A question turn ran
//            no stage and records none: it stays pending, and the card draws no timeline for it.
// A stage that is done or skipped is final in every turn. -> {state, ms?, detail?, skipped?}
export function stageState(view, name) {
  const recorded = view.stages[name] || { state: 'pending' };
  if (recorded.state !== 'running' && recorded.state !== 'pending') return recorded;
  if (view.status === 'aborted') return { state: 'skipped', skipped: 'stopped' };
  if (view.status === 'error') return recorded.state === 'running' ? { state: 'error' } : { state: 'skipped', skipped: 'not_run' };
  if (view.status === 'done' && recorded.state === 'pending' && Object.keys(view.stages).length) return { state: 'skipped', skipped: 'not_run' };
  return recorded;
}

// True while the labels are still to come: the turn is running and its label stage has not ended (done or skipped). labels is the server's
// own word (/v1/models features.labels): false means it runs no labeller, so no labels are coming and none are pending (P6-C).
export const labelsPending = (view, labels = true) => labels !== false && view.status === 'running'
  && !['done', 'skipped'].includes(view.stages.label?.state);
