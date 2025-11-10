package main

import (
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"
)

var (
	// Project Gutenberg START marker
	headerRe = regexp.MustCompile(`(?i)\*\*\*\s*START OF (THIS|THE) PROJECT GUTENBERG EBOOK.*`)
	// Project Gutenberg END marker
	footerRe = regexp.MustCompile(`(?i)\*\*\*\s*END OF (THIS|THE) PROJECT GUTENBERG EBOOK.*`)
)

// stripGutenbergBoilerplate removes content outside the Project Gutenberg START/END
// markers. If markers are not found, it returns the input unchanged.
func stripGutenbergBoilerplate(s string) string {
	lines := strings.Split(s, "\n")
	start := 0
	end := len(lines)
	for i, ln := range lines {
		if headerRe.MatchString(ln) {
			start = i + 1
			break
		}
	}
	for i := len(lines) - 1; i >= 0; i-- {
		if footerRe.MatchString(lines[i]) {
			end = i
			break
		}
	}
	if start == 0 && end == len(lines) {
		return s
	}
	return strings.Join(lines[start:end], "\n")
}

// normalizeNewlines converts CRLF and CR newlines to LF.
func normalizeNewlines(s string) string {
	s = strings.ReplaceAll(s, "\r\n", "\n")
	s = strings.ReplaceAll(s, "\r", "\n")
	return s
}

// RunClean processes exactly one raw input matching data/raw/*_raw.txt by removing
// Project Gutenberg boilerplate, normalizing newlines to LF, trimming space, and
// writing data/interim/<base>_clean.txt. It fails clearly if 0 or >1 inputs match.
func RunClean() error {
	name, err := discoverSingle(dirRaw, sufRaw)
	if err != nil {
		return fmt.Errorf("clean: %w", err)
	}
	return cleanFile(name)
}

func cleanFile(file string) error {
	inPath := filepath.Join(dirRaw, filepath.Base(file))
	if _, err := os.Stat(inPath); err != nil {
		return fmt.Errorf("clean: cannot stat input %s: %w", inPath, err)
	}

	b, err := os.ReadFile(inPath)
	if err != nil {
		return fmt.Errorf("clean: read %s: %w", inPath, err)
	}

	base := deriveBase(file, sufRaw)

	s := string(b)
	s = normalizeNewlines(s)
	s = stripGutenbergBoilerplate(s)
	s = strings.TrimSpace(s)

	outDir := dirInterim
	if err := os.MkdirAll(outDir, 0o755); err != nil {
		return fmt.Errorf("clean: mkdir %s: %w", outDir, err)
	}
	outPath := filepath.Join(outDir, base+sufClean)
	if err := os.WriteFile(outPath, []byte(s), 0644); err != nil {
		return fmt.Errorf("clean: write %s: %w", outPath, err)
	}
	fmt.Println("wrote", outPath)
	return nil
}
