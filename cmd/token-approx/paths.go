package main

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

const (
	dirRaw      = "data/raw"
	dirInterim  = "data/interim"
	dirSamples  = "data/processed/samples"
	dirDatasets = "data/processed/datasets"
	sufRaw      = "_raw.txt"
	sufClean    = "_clean.txt"
)

func deriveBase(file, suffix string) string {
	b := filepath.Base(file)
	b = strings.TrimSuffix(b, suffix)
	return b
}

// discoverSingle finds exactly one regular, non-hidden file in dir that ends with suffix and returns its filename.
func discoverSingle(dir, suffix string) (string, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		if os.IsNotExist(err) {
			return "", fmt.Errorf("no input files found in %s matching *%s", dir, suffix)
		}
		return "", fmt.Errorf("readdir %s: %w", dir, err)
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
		if strings.HasSuffix(name, suffix) {
			candidates = append(candidates, name)
		}
	}
	if len(candidates) == 0 {
		return "", fmt.Errorf("no input files found in %s matching *%s", dir, suffix)
	}
	if len(candidates) > 1 {
		return "", fmt.Errorf("multiple input files found in %s matching *%s", dir, suffix)
	}
	return candidates[0], nil
}
