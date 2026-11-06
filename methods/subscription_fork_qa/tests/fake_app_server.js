// Local protocol fixture; does not contact a model or network.
const fs = require('fs');
const readline = require('readline');
if (process.argv.includes('--version')) {
  process.stdout.write('codex-cli 0.167.0\n');
  process.exit(0);
}
let forks = 0;
let turns = 0;
function send(x) { process.stdout.write(JSON.stringify(x) + '\n'); }
readline.createInterface({input: process.stdin}).on('line', line => {
  const req = JSON.parse(line);
  if (!req.id) return;
  fs.appendFileSync(process.env.BVB_TEST_TRACE, JSON.stringify(req) + '\n');
  const reply = result => send({jsonrpc: '2.0', id: req.id, result});
  if (req.method === 'initialize') return reply({});
  if (req.method === 'thread/start') return reply({thread: {id: 'parent'}});
  if (req.method === 'thread/fork') return reply({thread: {id: `child-${++forks}`}});
  if (req.method === 'turn/start') {
    const turnId = `turn-${++turns}`;
    const failed = process.env.BVB_TEST_FAIL_CHILD === '1' && turns === 2;
    reply({turn: {id: turnId}});
    send({method: 'item/completed', params: {turnId, item: {type: 'agentMessage', text: failed ? '' : `answer-${turns}`}}});
    send({method: 'thread/tokenUsage/updated', params: {turnId, tokenUsage: {last: {inputTokens: 20, cachedInputTokens: 10, outputTokens: 2}}}});
    send({method: 'turn/completed', params: {turn: {id: turnId, status: failed ? 'failed' : 'completed'}}});
  }
});
