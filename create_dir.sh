#!/bin/bash

# Default values
day_num=1
base_dir="."

# Process command line arguments
while [ "$#" -gt 0 ]; do
  case "$1" in
    --days)
      day_num="$2"
      shift 2
      ;;
    --dir)
      base_dir="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--days <number>] [--dir <base_directory>]"
      exit 1
      ;;
  esac
done

# Create directory name with padding
day_name=$(printf "day%03d" $day_num)
day_path="$base_dir/$day_name"

echo "Creating directory: $day_path"
mkdir -p "$day_path"

# Create README.md template
cat > "$day_path/README.md" << EOF
# Day $day_num - CUDA Learning

## Book Coverage
Chapter X - [Chapter Title]

## Concepts Learned
- [Concept 1]
- [Concept 2]
- [Concept 3]

## Code Implemented
- [Description of implementations]

## Key Insights
- [Important takeaways]

## Challenges
- [Any challenges faced]

## Notes
[Additional notes or observations]
EOF

# Create Makefile with the specified format and TODO comments
cat > "$day_path/Makefile" << 'EOF'
# TODO: Replace SOURCE with your actual CUDA source file
NVCC = nvcc
NVCCFLAGS = -std=c++11 -O3
TARGET = target # TODO: Replace with your target executable name
SOURCE = source.cu # TODO: Replace with your source file

all: $(TARGET)

$(TARGET): $(SOURCE)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f $(TARGET)

run:
	./$(TARGET)

.PHONY: all clean run
EOF

# Create .gitignore
cat > "$day_path/.gitignore" << 'EOF'
# Compiled Object files
*.slo
*.lo
*.o
*.obj

# Executables
*.exe
*.out
*.app

# CUDA specific
*.i
*.ii
*.gpu
*.ptx
*.cubin
*.fatbin

# Editor files
.vscode/
.idea/
*.swp
*~

# OS specific
.DS_Store
Thumbs.db
EOF

echo "Successfully created $day_path with README.md, Makefile, and .gitignore"
echo "Note: Need to manually create your CUDA source files and update the Makefile accordingly."
