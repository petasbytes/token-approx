// Command token-approx prepares text data for token approximation experiments.
// It provides subcommands to download sample text, clean boilerplate, split
// into deterministic samples, and measure token counts, reading and writing
// under ./data relative to the current working directory.
package main

import (
	"os"

	"github.com/spf13/cobra"
)

func main() {
	rootCmd := &cobra.Command{
		Use:          "token-approx",
		Short:        "Prepare text data for token approximation experiments",
		SilenceUsage: true,
		CompletionOptions: cobra.CompletionOptions{
			DisableDefaultCmd: true,
		},
	}

	getDataCmd := &cobra.Command{
		Use:     "get-data",
		Short:   "Download example text into data/raw/oliver-twist_gberg_raw.txt",
		Example: `  token-approx get-data`,
		Args:    cobra.NoArgs,
		RunE: func(cmd *cobra.Command, args []string) error {
			return RunGetData()
		},
	}
	rootCmd.AddCommand(getDataCmd)

	cleanCmd := &cobra.Command{
		Use:     "clean",
		Short:   "Clean raw file into data/interim/<basename>_clean.txt",
		Example: `  token-approx clean`,
		Args:    cobra.NoArgs,
		RunE: func(cmd *cobra.Command, args []string) error {
			return RunClean()
		},
	}
	rootCmd.AddCommand(cleanCmd)

	splitCmd := &cobra.Command{
		Use:     "split",
		Short:   "Split clean file into data/processed/samples",
		Example: `  token-approx split`,
		Args:    cobra.NoArgs,
		RunE: func(cmd *cobra.Command, args []string) error {
			return RunSplit()
		},
	}
	rootCmd.AddCommand(splitCmd)

	measureCmd := &cobra.Command{
		Use:     "measure",
		Short:   "Measure features, get token counts, and append JSONL records to data/processed/datasets/dataset.jsonl",
		Example: `  token-approx measure`,
		Args:    cobra.NoArgs,
		RunE: func(cmd *cobra.Command, args []string) error {
			return RunMeasure()
		},
	}
	rootCmd.AddCommand(measureCmd)

	if err := rootCmd.Execute(); err != nil {
		os.Exit(1)
	}
}
