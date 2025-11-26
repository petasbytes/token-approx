#!/usr/bin/env node
'use strict';

const fs = require('fs');

let tokenizer;
try {
  tokenizer = require('@anthropic-ai/tokenizer');
} catch (e) {
  console.error("Failed to load '@anthropic-ai/tokenizer'. Install it in this project: `yarn add @anthropic-ai/tokenizer` or `npm i @anthropic-ai/tokenizer`.\n" + String(e && e.message ? e.message : e));
  process.exit(1);
}

function count(text) {
  // The README recommends: import { countTokens } from '@anthropic-ai/tokenizer'
  // Handle both named export and potential default wrappers conservatively.
  if (tokenizer && typeof tokenizer.countTokens === 'function') return tokenizer.countTokens(text);
  if (tokenizer && tokenizer.default && typeof tokenizer.default.countTokens === 'function') return tokenizer.default.countTokens(text);
  throw new Error('countTokens export not found on @anthropic-ai/tokenizer');
}

function main() {
  const args = process.argv.slice(2);
  args.forEach((p) => {
    try {
      const txt = fs.readFileSync(p, { encoding: 'utf8' });
      const n = count(txt);
      process.stdout.write(JSON.stringify({ source_path: p, pred_tokens: Number(n) }) + '\n');
    } catch (e) {
      process.stdout.write(
        JSON.stringify({ source_path: p, error: String((e && e.message) || e) }) + '\n'
      );
    }
  });
}

main();
