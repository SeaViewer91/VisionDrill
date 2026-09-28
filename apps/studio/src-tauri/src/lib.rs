// 모든 로직은 웹뷰(TypeScript + sql.js)에 있고, Rust 쪽은 파일 대화상자·파일시스템 플러그인만 연결한다.
#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())

        .run(tauri::generate_context!())
        .expect("error while running VisionDrill Studio");
}
