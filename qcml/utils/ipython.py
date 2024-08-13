from IPython.core.magic import register_line_magic
from IPython.display import display, HTML

@register_line_magic
def scroll_output(line):
    """Magic command to enable scrolling in output cells with a specified max height."""
    height = line.strip() or "300px"
    display(HTML(f'''
        <style>
            .output_area pre {{
                max-height: {height};
                overflow-y: scroll;
            }}
            .output_area {{
                max-height: {height};
                overflow-y: scroll;
            }}
        </style>
    '''))

# Now you can use the magic command in your notebook
