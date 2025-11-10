package main

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/anthropics/anthropic-sdk-go"
)

type features struct {
	Bytes int `json:"bytes"`
	Runes int `json:"runes"`
	Words int `json:"words"`
	Lines int `json:"lines"`
}

type record struct {
	ID          string   `json:"id"`
	Model       string   `json:"model"`
	InputTokens int64    `json:"input_tokens"`
	Features    features `json:"features"`
	SourcePath  string   `json:"source_path"`
}

type tokenCounter interface {
	Count(ctx context.Context, params anthropic.MessageCountTokensParams) (int64, error)
}

type sdkTokenCounter struct {
	client anthropic.Client
}

const defaultModel = anthropic.ModelClaude3_7SonnetLatest

func countFeatures(s string) features {
	b := len(s)
	r := utf8.RuneCountInString(s)
	w := len(strings.Fields(s))
	l := 0
	if s != "" {
		l = 1 + strings.Count(s, "\n")
	}
	return features{Bytes: b, Runes: r, Words: w, Lines: l}
}

// CountInputTokens builds a single-user-message request and counts tokens
// via Anthropic Messages.CountTokens. Abstracted behind tokenCounter for testability.
func CountInputTokens(ctx context.Context, s string) (int64, error) {
	params := anthropic.MessageCountTokensParams{
		Model: defaultModel,
		Messages: []anthropic.MessageParam{
			anthropic.NewUserMessage(anthropic.NewTextBlock(s)),
		},
	}
	return currentTokenCounter.Count(ctx, params)
}

// Count adapts to anthropic-sdk-go, invoking Messages.CountTokens and returning InputTokens.
func (s sdkTokenCounter) Count(ctx context.Context, params anthropic.MessageCountTokensParams) (int64, error) {
	count, err := s.client.Messages.CountTokens(ctx, params)
	if err != nil {
		return 0, err
	}
	return count.InputTokens, nil
}

var currentTokenCounter tokenCounter = sdkTokenCounter{client: anthropic.NewClient()}

// RunMeasure batches over samples and appends JSONL records. It computes local features
// and calls the Anthropic Count Tokens API for input_tokens (requires ANTHROPIC_API_KEY).
func RunMeasure() error {
	// Early exit if API key missing
	if os.Getenv("ANTHROPIC_API_KEY") == "" {
		return fmt.Errorf("missing ANTHROPIC_API_KEY")
	}

	// Enumerate candidate sample files (non-recursive), regular, non-hidden, *.txt
	entries, err := os.ReadDir(dirSamples)
	if err != nil {
		if os.IsNotExist(err) {
			return fmt.Errorf("measure: no samples present in %s", dirSamples)
		}
		return fmt.Errorf("measure: readdir %s: %w", dirSamples, err)
	}
	var candidates []string
	for _, e := range entries {
		name := e.Name()
		if strings.HasPrefix(name, ".") {
			continue
		}
		if !e.Type().IsRegular() {
			continue
		}
		if !strings.HasSuffix(strings.ToLower(name), ".txt") {
			continue
		}
		candidates = append(candidates, name)
	}

	N_total := len(candidates)
	if N_total == 0 {
		return fmt.Errorf("measure: no samples present in %s", dirSamples)
	}
	sort.Strings(candidates)

	// Build seen set from existing JSONL (if present)
	seen := make(map[string]struct{})
	outDir := dirDatasets
	outPath := filepath.Join(outDir, "dataset.jsonl")
	if f, err := os.Open(outPath); err == nil {
		scanner := bufio.NewScanner(f)
		for scanner.Scan() {
			var rec struct {
				SourcePath string `json:"source_path"`
			}
			if err := json.Unmarshal(scanner.Bytes(), &rec); err == nil && rec.SourcePath != "" {
				seen[rec.SourcePath] = struct{}{}
			}
		}
		_ = f.Close()
	}

	// Determine files to process (repo-root-relative paths)
	var toProcess []string
	N_skipped := 0
	for _, name := range candidates {
		rel := filepath.Join(dirSamples, name)
		if _, ok := seen[rel]; ok {
			N_skipped++
			continue
		}
		toProcess = append(toProcess, rel)
	}

	N_toProcess := len(toProcess)
	if N_toProcess == 0 {
		fmt.Println("No new files to process. To re-process a specific file, delete its entry in data/processed/datasets/dataset.jsonl and rerun token-approx measure.")
		return nil
	}

	if err := os.MkdirAll(outDir, 0o755); err != nil {
		return fmt.Errorf("measure: mkdir %s: %w", outDir, err)
	}
	f, err := os.OpenFile(outPath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return fmt.Errorf("measure: open %s for append: %w", outPath, err)
	}
	defer f.Close()

	// Process each file with token counting and local features
	var hadFailure bool
	N_success := 0
	N_failed := 0
	var failedPaths []string
	for _, rel := range toProcess {
		b, err := os.ReadFile(rel)
		if err != nil {
			hadFailure = true
			N_failed++
			failedPaths = append(failedPaths, rel)
			continue
		}

		ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
		inputTokens, err := CountInputTokens(ctx, string(b))
		cancel()
		if err != nil {
			hadFailure = true
			N_failed++
			failedPaths = append(failedPaths, rel)
			continue
		}

		now := time.Now().UTC()
		id := fmt.Sprintf("rec-%s_%s_%09d", now.Format("20060102"), now.Format("150405"), now.Nanosecond())
		feats := countFeatures(string(b))
		rec := record{
			ID:          id,
			Model:       string(defaultModel),
			InputTokens: inputTokens,
			Features:    feats,
			SourcePath:  rel,
		}
		enc, err := json.Marshal(rec)
		if err != nil {
			hadFailure = true
			N_failed++
			failedPaths = append(failedPaths, rel)
			continue
		}
		if _, err := f.Write(append(enc, '\n')); err != nil {
			hadFailure = true
			N_failed++
			failedPaths = append(failedPaths, rel)
			continue
		}
		N_success++
	}

	fmt.Println("Files in directory:", N_total)
	fmt.Println("Already present (skipped):", N_skipped)
	fmt.Println("Attempted (new):", N_toProcess)
	fmt.Println("Processed successfully:", N_success)
	fmt.Println("Failed:", N_failed)
	if N_failed > 0 {
		fmt.Println("Failed files:", strings.Join(failedPaths, ", "))
	}

	if hadFailure {
		return fmt.Errorf("measure: some files failed to process")
	}
	return nil
}
