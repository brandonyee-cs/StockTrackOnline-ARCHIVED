document.getElementById('loginForm').addEventListener('submit', function(e) {
    e.preventDefault();
    var username = document.getElementById('username').value;
    var password = document.getElementById('password').value;
    // Add form validation here
    fetch('/login', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: new URLSearchParams({
            'username': username,
            'password': password
        })
    }).then(function(response) {
        return response.text();
    }).then(function(text) {
        console.log(text);
    });
});