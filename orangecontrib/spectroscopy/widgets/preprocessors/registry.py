class PreprocessorEditorRegistry:

    def __init__(self):
        self.registered = []

    def register(self, editor, priority=1000):
        self.registered.append((editor, priority))

    def sorted(self):
        for editor, _ in sorted(self.registered, key=lambda x: x[1]):
            yield editor


preprocess_editors = PreprocessorEditorRegistry()

emsc_editors = PreprocessorEditorRegistry()
