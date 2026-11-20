/* Keep the sidebar sections a reader opened open when they go to another page. */
(function () {
  var KEY = "diffractix-nav-open";
  var toggles = document.querySelectorAll(".md-sidebar--primary .md-nav__toggle");
  var saved = [];
  try {
    saved = JSON.parse(sessionStorage.getItem(KEY)) || [];
  } catch (error) {}

  function save() {
    var open = [];
    toggles.forEach(function (toggle) {
      if (toggle.checked) open.push(toggle.id);
    });
    try {
      sessionStorage.setItem(KEY, JSON.stringify(open));
    } catch (error) {}
  }

  toggles.forEach(function (toggle) {
    if (saved.indexOf(toggle.id) !== -1) toggle.checked = true;
    toggle.addEventListener("change", save);
  });
  save();
})();
