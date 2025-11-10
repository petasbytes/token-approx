package main

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// chunkText splits s into rune-based chunks, preferring the last "\n\n" between
// target and max; otherwise it cuts at max. Chunks are trimmed.
func chunkText(s string, target, max int) []string {
	if target <= 0 || max <= 0 || max < target {
		return []string{strings.TrimSpace(s)}
	}
	rs := []rune(s)
	var samples []string
	i := 0
	n := len(rs)
	for i < n {
		j := i + target
		if j > n {
			j = n
		}
		end := i + max
		if end > n {
			end = n
		}
		// Search only within [j, end) for the last "\n\n" boundary.
		k := -1
		if j < end {
			// Scan backwards to find two consecutive '\n'.
			for p := end - 1; p >= j; p-- {
				// We need p-1 >= j to have two runes within [j,end)
				if p-1 >= j && rs[p-1] == '\n' && rs[p] == '\n' {
					k = p - 1 // index of the first '\n' in the pair
					break
				}
			}
		}
		if k == -1 || k <= i {
			if end > i {
				k = end
			} else {
				k = n
			}
		}
		sample := strings.TrimSpace(string(rs[i:k]))
		samples = append(samples, sample)
		i = k
	}
	return samples
}

// splitCleanFile chunks one <base>_clean.txt and writes <base>_sample-XXX.txt files.
func splitCleanFile(file string) error {
	inPath := filepath.Join(dirInterim, filepath.Base(file))
	if _, err := os.Stat(inPath); err != nil {
		return fmt.Errorf("split: cannot stat input %s: %w", inPath, err)
	}

	b, err := os.ReadFile(inPath)
	if err != nil {
		return fmt.Errorf("split: read %s: %w", inPath, err)
	}

	base := deriveBase(file, sufClean)

	samples := chunkText(string(b), 32000, 36000)
	outDir := dirSamples
	if err := os.MkdirAll(outDir, 0o755); err != nil {
		return fmt.Errorf("split: mkdir %s: %w", outDir, err)
	}
	for i, s := range samples {
		if len(s) == 0 {
			continue
		}
		outPath := filepath.Join(outDir, fmt.Sprintf("%s_sample-%03d.txt", base, i+1))
		if err := os.WriteFile(outPath, []byte(s), 0o644); err != nil {
			return fmt.Errorf("split: write %s: %w", outPath, err)
		}
		fmt.Println("wrote", outPath)
	}
	return nil
}

// RunSplit splits exactly one data/interim/*_clean.txt into numbered samples
// under data/processed/samples; returns an error if 0 or >1 inputs exist.
func RunSplit() error {
	name, err := discoverSingle(dirInterim, sufClean)
	if err != nil {
		return fmt.Errorf("split: %w", err)
	}
	return splitCleanFile(name)
}
