type ImportButtonsProps = {
  cols: number;
  gridSize: number;
  onBigger: () => void;
  onSmaller: () => void;
  setGridPreview: (checked: boolean) => void;
  gridPreview: boolean;
  selectedClips: Set<string>;
  setSelectedClips: React.Dispatch<
    React.SetStateAction<Set<string>>
  >;
  onImport: () => void;
  loading: boolean;
  clips: { id: string }[];
};

export default function ImportButtons(props: ImportButtonsProps) {
  const allSelected = props.clips.length > 0 && props.selectedClips.size === props.clips.length;
  const someSelected = props.selectedClips.size > 0 && !allSelected;

  const handleToggleSelectAll = () => {
    if (allSelected) {
      props.setSelectedClips(new Set());
    } else {
      props.setSelectedClips(new Set(props.clips.map((c) => c.id)));
    }
  };

  return (
    <main className="clips-import">
      <div className="import-buttons-container">
        <button
          onClick={() => {
            props.onImport();
          }}
          disabled={props.loading}
          id="file-button"
        >
          {props.loading ? "Processing..." : "Import Episode"}
        </button>
      </div>
      <div className="grid-checkboxes">
        <div className="selectable-checkboxes">
          <div className="checkbox-row">
            <label className="custom-checkbox">
              <input
                type="checkbox"
                className="checkbox"
                checked={props.gridPreview}
                onChange={(e) => props.setGridPreview(e.target.checked)}
              />
              <span className="checkmark"></span>
            </label>
            <span>Grid preview</span>
          </div>
          <div className="checkbox-row">
            <label className="custom-checkbox">
              <input
                type="checkbox"
                className="checkbox"
                checked={allSelected || someSelected}
                disabled={props.clips.length === 0}
                onChange={handleToggleSelectAll}
              />
              <span className={`checkmark ${someSelected ? "partial" : ""}`}></span>
            </label>
            <span>
              Select ({props.selectedClips.size})
            </span>
          </div>
        </div>
        <div className="zoomWrapper">
          <span>Size: {props.gridSize}px</span>
          <form>
            <button type="button" onClick={props.onSmaller}>
              -
            </button>
            <button type="button" onClick={props.onBigger}>
              +
            </button>
          </form>
        </div>
      </div>
    </main>
  );
}