package main

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthropics/anthropic-sdk-go"
)

// chdir changes directory and registers a cleanup to restore.
func chdir(t *testing.T, dir string) {
	t.Helper()
	old, err := os.Getwd()
	if err != nil {
		t.Fatalf("getwd: %v", err)
	}
	if err := os.Chdir(dir); err != nil {
		t.Fatalf("chdir %s: %v", dir, err)
	}
	t.Cleanup(func() { _ = os.Chdir(old) })
}

// writeFile ensures parent dirs and writes a file with given contents.
func writeFile(t *testing.T, p, s string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		t.Fatalf("mkdir %s: %v", filepath.Dir(p), err)
	}
	if err := os.WriteFile(p, []byte(s), 0o644); err != nil {
		t.Fatalf("write %s: %v", p, err)
	}
}

// countLines counts trailing-\n separated lines in a byte slice.
func countLines(b []byte) int {
	c := 0
	for _, ch := range b {
		if ch == '\n' {
			c++
		}
	}
	return c
}

type fakeCounterSuccess struct{ tokens int64 }

func (f fakeCounterSuccess) Count(ctx context.Context, params anthropic.MessageCountTokensParams) (int64, error) {
	return f.tokens, nil
}

// fakeErrorCounter simulates per-file failures by inspecting message content.
// If the user message text contains the substring failIfContains, returns an error.
type fakeCounterErrorOnMarker struct {
	tokens int64
	marker string
}

func (f fakeCounterErrorOnMarker) Count(ctx context.Context, params anthropic.MessageCountTokensParams) (int64, error) {
	b, _ := json.Marshal(params)
	if f.marker != "" && strings.Contains(string(b), f.marker) {
		return 0, os.ErrDeadlineExceeded
	}
	return f.tokens, nil
}

// -------- Clean subcommand --------

func TestClean(t *testing.T) {
	t.Run("No raw input file", func(t *testing.T) {
		dir := t.TempDir()
		chdir(t, dir)
		// No matching data/raw/*_raw.txt present
		err := RunClean()
		if err == nil {
			t.Fatalf("expected error for missing input")
		}
	})

	t.Run("Multiple raw input files", func(t *testing.T) {
		dir := t.TempDir()
		chdir(t, dir)
		writeFile(t, "data/raw/a_raw.txt", "A")
		writeFile(t, "data/raw/b_raw.txt", "B")
		if err := RunClean(); err == nil {
			t.Fatalf("expected error when multiple raw inputs exist")
		}
	})

	t.Run("Produces cleaned output with correct suffix", func(t *testing.T) {
		dir := t.TempDir()
		chdir(t, dir)
		writeFile(t, "data/raw/oliver-twist_gberg_raw.txt", "*** START OF THIS PROJECT GUTENBERG EBOOK\nBody\n*** END OF THIS PROJECT GUTENBERG EBOOK\n")
		if err := RunClean(); err != nil {
			t.Fatalf("RunClean: %v", err)
		}
		if _, err := os.Stat("data/interim/oliver-twist_gberg_clean.txt"); err != nil {
			t.Fatalf("expected output file to exist: %v", err)
		}
		b, err := os.ReadFile("data/interim/oliver-twist_gberg_clean.txt")
		if err != nil {
			t.Fatalf("read cleaned file: %v", err)
		}
		if len(b) == 0 {
			t.Fatalf("expected cleaned output to be non-empty")
		}
	})
}

// -------- Split subcommand --------

func TestSplit(t *testing.T) {
	t.Run("No clean input file", func(t *testing.T) {
		dir := t.TempDir()
		chdir(t, dir)
		// No matching data/interim/*_clean.txt present
		err := RunSplit()
		if err == nil {
			t.Fatalf("expected error for missing input")
		}
	})

	t.Run("Multiple clean input files", func(t *testing.T) {
		dir := t.TempDir()
		chdir(t, dir)
		writeFile(t, "data/interim/a_clean.txt", strings.Repeat("x", 100))
		writeFile(t, "data/interim/b_clean.txt", strings.Repeat("y", 100))
		if err := RunSplit(); err == nil {
			t.Fatalf("expected error when multiple clean inputs exist")
		}
	})

	t.Run("Produces samples with correct suffix", func(t *testing.T) {
		dir := t.TempDir()
		chdir(t, dir)
		writeFile(t, "data/interim/oliver-twist_gberg_clean.txt", strings.Repeat("para\n\n", 20000))
		if err := RunSplit(); err != nil {
			t.Fatalf("RunSplit: %v", err)
		}
		if _, err := os.Stat("data/processed/samples/oliver-twist_gberg_sample-001.txt"); err != nil {
			t.Fatalf("expected sample-001 to exist: %v", err)
		}
	})
}

// -------- Measure subcommand --------

func TestMeasure(t *testing.T) {
	t.Run("Missing API key prevents any work", func(t *testing.T) {
		dir := t.TempDir()
		chdir(t, dir)
		writeFile(t, "data/processed/samples/oliver-twist_gberg_sample-001.txt", "content")
		// Ensure key is not set
		t.Setenv("ANTHROPIC_API_KEY", "")
		out := "data/processed/datasets/dataset.jsonl"
		err := RunMeasure()
		if err == nil {
			t.Fatalf("expected error when API key missing")
		}
		if _, statErr := os.Stat(out); !os.IsNotExist(statErr) {
			t.Fatalf("expected %s to not exist when API key missing", out)
		}
	})

	t.Run("No samples present", func(t *testing.T) {
		dir := t.TempDir()
		chdir(t, dir)
		t.Setenv("ANTHROPIC_API_KEY", "dummy")
		if err := os.MkdirAll("data/processed/samples", 0o755); err != nil {
			t.Fatalf("mkdir samples: %v", err)
		}
		// Stub token counter (no network)
		old := currentTokenCounter
		currentTokenCounter = fakeCounterSuccess{tokens: 1}
		t.Cleanup(func() { currentTokenCounter = old })
		err := RunMeasure()
		if err == nil {
			t.Fatalf("expected error when no samples present")
		}
		if _, err := os.Stat("data/processed/datasets/dataset.jsonl"); !os.IsNotExist(err) {
			t.Fatalf("expected dataset.jsonl to not exist when no samples present")
		}
	})

	t.Run("Ignores hidden and non-regular files", func(t *testing.T) {
		dir := t.TempDir()
		chdir(t, dir)
		t.Setenv("ANTHROPIC_API_KEY", "dummy")
		// Inputs: hidden files, dir, and one regular
		writeFile(t, "data/processed/samples/.DS_Store", "")
		writeFile(t, "data/processed/samples/.hidden.txt", "secret")
		if err := os.MkdirAll("data/processed/samples/subdir", 0o755); err != nil {
			t.Fatalf("mkdir subdir: %v", err)
		}
		writeFile(t, "data/processed/samples/subdir/ignored.txt", "child")
		writeFile(t, "data/processed/samples/basename_sample-001.txt", "visible content")
		// Stub token counter
		old := currentTokenCounter
		currentTokenCounter = fakeCounterSuccess{tokens: 5}
		t.Cleanup(func() { currentTokenCounter = old })
		if err := RunMeasure(); err != nil {
			t.Fatalf("RunMeasure: %v", err)
		}
		b, err := os.ReadFile("data/processed/datasets/dataset.jsonl")
		if err != nil {
			t.Fatalf("read dataset.jsonl: %v", err)
		}
		if countLines(b) != 1 {
			t.Fatalf("expected exactly 1 JSONL line, got %d", countLines(b))
		}
	})

	t.Run("Creates JSONL when missing; appends on later run", func(t *testing.T) {
		dir := t.TempDir()
		chdir(t, dir)
		t.Setenv("ANTHROPIC_API_KEY", "dummy")
		writeFile(t, "data/processed/samples/oliver-twist_gberg_sample-001.txt", "hello world")
		writeFile(t, "data/processed/samples/oliver-twist_gberg_sample-002.txt", "hello again")
		// Stub token counter
		old := currentTokenCounter
		currentTokenCounter = fakeCounterSuccess{tokens: 42}
		t.Cleanup(func() { currentTokenCounter = old })
		if err := RunMeasure(); err != nil {
			t.Fatalf("RunMeasure: %v", err)
		}
		if _, err := os.Stat("data/processed/datasets/dataset.jsonl"); err != nil {
			t.Fatalf("expected dataset.jsonl to exist: %v", err)
		}
		b, err := os.ReadFile("data/processed/datasets/dataset.jsonl")
		if err != nil {
			t.Fatalf("read dataset.jsonl: %v", err)
		}
		if countLines(b) != 2 {
			t.Fatalf("expected 2 JSONL lines, got %d", countLines(b))
		}
		// Add a new sample and rerun: expect N+1 lines
		writeFile(t, "data/processed/samples/oliver-twist_gberg_sample-003.txt", "third")
		if err := RunMeasure(); err != nil {
			t.Fatalf("RunMeasure second run: %v", err)
		}
		b2, err := os.ReadFile("data/processed/datasets/dataset.jsonl")
		if err != nil {
			t.Fatalf("read dataset.jsonl (second): %v", err)
		}
		if countLines(b2) != countLines(b)+1 {
			t.Fatalf("expected N+1 JSONL lines after rerun, got %d then %d", countLines(b), countLines(b2))
		}
	})

	t.Run("Idempotent skip; only new files appended", func(t *testing.T) {
		dir := t.TempDir()
		chdir(t, dir)
		t.Setenv("ANTHROPIC_API_KEY", "dummy")
		// Seed three samples and pre-seed JSONL with 001 entry
		writeFile(t, "data/processed/samples/oliver-twist_gberg_sample-001.txt", "one")
		writeFile(t, "data/processed/samples/oliver-twist_gberg_sample-002.txt", "two")
		writeFile(t, "data/processed/samples/oliver-twist_gberg_sample-003.txt", "three")
		out := "data/processed/datasets/dataset.jsonl"
		// Pre-seed with an existing record referencing sample-001
		writeFile(t, out, "{\"source_path\":\"data/processed/samples/oliver-twist_gberg_sample-001.txt\"}\n")
		// Stub token counter
		old := currentTokenCounter
		currentTokenCounter = fakeCounterSuccess{tokens: 77}
		t.Cleanup(func() { currentTokenCounter = old })
		if err := RunMeasure(); err != nil {
			t.Fatalf("RunMeasure: %v", err)
		}
		b, err := os.ReadFile(out)
		if err != nil {
			t.Fatalf("read dataset.jsonl: %v", err)
		}
		if countLines(b) != 3 { // 1 pre-seeded + 2 new (002,003)
			t.Fatalf("expected 3 JSONL lines, got %d", countLines(b))
		}
		// Rerun with no new files: expect unchanged line count
		if err := RunMeasure(); err != nil {
			t.Fatalf("RunMeasure rerun: %v", err)
		}
		b2, err := os.ReadFile(out)
		if err != nil {
			t.Fatalf("read dataset.jsonl rerun: %v", err)
		}
		if countLines(b2) != countLines(b) {
			t.Fatalf("expected 0 new lines on rerun, got %d -> %d", countLines(b), countLines(b2))
		}
	})

	t.Run("Partial failures continue; exit is non-zero", func(t *testing.T) {
		dir := t.TempDir()
		chdir(t, dir)
		t.Setenv("ANTHROPIC_API_KEY", "dummy")
		writeFile(t, "data/processed/samples/sample-A.txt", "ok")
		writeFile(t, "data/processed/samples/sample-B.txt", "ERR will fail")
		writeFile(t, "data/processed/samples/sample-C.txt", "ok again")
		out := "data/processed/datasets/dataset.jsonl"
		old := currentTokenCounter
		currentTokenCounter = fakeCounterErrorOnMarker{tokens: 10, marker: "ERR"}
		t.Cleanup(func() { currentTokenCounter = old })
		err := RunMeasure()
		if err == nil {
			t.Fatalf("expected non-nil error when some files fail")
		}
		b, err2 := os.ReadFile(out)
		if err2 != nil {
			t.Fatalf("read dataset.jsonl: %v", err2)
		}
		if countLines(b) != 2 {
			t.Fatalf("expected lines for A and C only (2), got %d", countLines(b))
		}
	})
}
