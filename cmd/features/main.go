package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/anthropics/anthropic-sdk-go"
)

func main() {
	filePath := flag.String("file", "", "path to input file")
	flag.Parse()

	if *filePath == "" {
		fmt.Fprintln(os.Stderr, "missing required --file PATH")
		os.Exit(2)
	}
	if os.Getenv("ANTHROPIC_API_KEY") == "" {
		fmt.Fprintln(os.Stderr, "missing ANTHROPIC_API_KEY in environment")
		os.Exit(2)
	}

	// Validate and read input file
	st, statErr := os.Stat(*filePath)
	if statErr != nil {
		fmt.Fprintf(os.Stderr, "stat file: %v\n", statErr)
		os.Exit(1)
	}
	if !st.Mode().IsRegular() {
		fmt.Fprintf(os.Stderr, "input is not a regular file: %s\n", *filePath)
		os.Exit(1)
	}
	b, rerr := os.ReadFile(*filePath)
	if rerr != nil {
		fmt.Fprintf(os.Stderr, "read file: %v\n", rerr)
		os.Exit(1)
	}
	user := string(b)

	// Generate identifier
	now := time.Now().UTC()
	id := fmt.Sprintf("rec-%s_%s_%09d", now.Format("20060102"), now.Format("150405"), now.Nanosecond())

	f := countFeatures(user)

	outDir := filepath.Join("data", "processed", "features")
	if err := os.MkdirAll(outDir, 0o755); err != nil {
		fmt.Fprintf(os.Stderr, "mkdir %s: %v\n", outDir, err)
		os.Exit(1)
	}
	featuresPath := filepath.Join(outDir, "features.jsonl")
	featuresFile, err := os.OpenFile(featuresPath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		fmt.Fprintf(os.Stderr, "open %s: %v\n", featuresPath, err)
		os.Exit(1)
	}
	defer featuresFile.Close()

	// Send one-turn message (non-streaming) to Anthropic Messages API, small MaxTokens
	client := anthropic.NewClient()
	params := anthropic.MessageNewParams{
		Model:     anthropic.ModelClaude3_7SonnetLatest,
		MaxTokens: 16,
		Messages: []anthropic.MessageParam{
			anthropic.NewUserMessage(anthropic.NewTextBlock(user)),
		},
	}
	resp, err := client.Messages.New(context.Background(), params)
	if err != nil {
		fmt.Fprintf(os.Stderr, "anthropic send: %v\n", err)
		os.Exit(1)
	}

	// Write single combined record line
	sourcePath := *filePath
	rec := record{
		ID:          id,
		Model:       string(anthropic.ModelClaude3_7SonnetLatest),
		InputTokens: resp.Usage.InputTokens,
		Features:    f,
		SourcePath:  sourcePath,
	}
	if err := writeJSONL(featuresFile, rec); err != nil {
		fmt.Fprintf(os.Stderr, "write record: %v\n", err)
		os.Exit(1)
	}
}

// features holds minimal text features for an input file (with JSON order)
type features struct {
	Bytes int `json:"bytes"`
	Runes int `json:"runes"`
	Words int `json:"words"`
	Lines int `json:"lines"`
}

// record is the JSON-encoded ordering for the combined record line.
type record struct {
	ID          string   `json:"id"`
	Model       string   `json:"model"`
	InputTokens int64    `json:"input_tokens"`
	Features    features `json:"features"`
	SourcePath  string   `json:"source_path"`
}

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

func writeJSONL(w io.Writer, v any) error {
	enc, err := json.Marshal(v)
	if err != nil {
		return err
	}
	if _, err := w.Write(append(enc, '\n')); err != nil {
		return err
	}
	return nil
}
