export const meta = {
  name: 'sonnet-cot-soundness',
  description: 'Independent Sonnet judge of CoT reasoning soundness (no gold shown) vs top_p, MMMU-Pro/Qwen3.5-9B',
  phases: [{ title: 'Judge', detail: 'one Sonnet subagent per (question, top_p) CoT: read image + trace -> sound?' }],
}
const A = (typeof args === 'string') ? JSON.parse(args) : (args || {})
const DIR = A.dir || '/root/eidos/mmmu_pro/qwen35_9b_cot_diversity/outputs/sonnet_judge'
const N = A.n || 200
log(`judging ${N} CoTs from ${DIR}`)
const V = { type: 'object', additionalProperties: false,
  properties: { id: { type: 'string' }, top_p: { type: 'number' },
    sound: { type: 'boolean' }, reason: { type: 'string' } },
  required: ['id', 'top_p', 'sound', 'reason'] }
phase('Judge')
const results = await parallel(Array.from({ length: N }, (_, i) => () =>
  agent(
    `You are a STRICT grader of visual reasoning. Read the task file at ${DIR}/task_${i}.txt — it ` +
    `contains an ID, a TOPP value, a multiple-choice QUESTION, its OPTIONS, one or more IMAGE FILE ` +
    `path(s), and a step-by-step REASONING TRACE from another model. Use the Read tool to open the ` +
    `image file(s) named in the task so you can verify what the trace claims to see. You are NOT ` +
    `told the correct answer — judge ONLY whether the reasoning is SOUND on its own merits.\n\n` +
    `SOUND requires BOTH: (1) every fact it reads off the image is accurate (no misread value, ` +
    `label, axis, shape, connection, or count), AND (2) every inferential/mathematical step is ` +
    `valid. If it misreads the image even once in a way that matters, OR makes an invalid step, it ` +
    `is UNSOUND — even if it reaches a plausible answer. A truncated/unfinished trace is UNSOUND ` +
    `unless what it produced is fully correct and already determines the answer.\n\n` +
    `Return the ID and TOPP exactly as written in the file, sound=true/false, and a one-sentence reason.`,
    { schema: V, model: 'sonnet', label: `task_${i}`, phase: 'Judge' }
  )
))
const ok = results.filter(Boolean)
const byp = {}
for (const r of ok) { (byp[r.top_p] = byp[r.top_p] || { s: 0, n: 0 }); byp[r.top_p].n++; if (r.sound) byp[r.top_p].s++ }
const summary = Object.keys(byp).map(Number).sort((a, b) => a - b)
  .map(p => ({ top_p: p, soundness: +(byp[p].s / byp[p].n).toFixed(3), n: byp[p].n }))
log('Sonnet soundness vs top_p: ' + JSON.stringify(summary))
return { summary, n_judged: ok.length, verdicts: ok }
