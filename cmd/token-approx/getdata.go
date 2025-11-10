package main

import (
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"time"
)

// RunGetData downloads the example Oliver Twist (Project Gutenberg) text
// into data/raw/oliver-twist_gberg_raw.txt.
func RunGetData() error {
	const url = "https://www.gutenberg.org/ebooks/730.txt.utf-8"
	client := &http.Client{Timeout: 15 * time.Second}
	req, err := http.NewRequest(http.MethodGet, url, nil)
	if err != nil {
		return err
	}
	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("unexpected status: %s", resp.Status)
	}
	b, err := io.ReadAll(resp.Body)
	if err != nil {
		return err
	}
	outDir := dirRaw
	if err := os.MkdirAll(outDir, 0o755); err != nil {
		return err
	}
	outPath := filepath.Join(outDir, "oliver-twist_gberg_raw.txt")
	if err := os.WriteFile(outPath, b, 0o644); err != nil {
		return fmt.Errorf("get-data: write %s: %w", outPath, err)
	}
	fmt.Println("wrote", outPath)
	return nil
}
