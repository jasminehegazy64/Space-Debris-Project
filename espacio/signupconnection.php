
<?php

if (empty($_POST["fn"])) {
    die("Name is required");
}
if (empty($_POST["ln"])) {
    die("Name is required");
}

if (empty($_POST["age"])) {
    die("Age is required");
}

if ( ! filter_var($_POST["email"], FILTER_VALIDATE_EMAIL)) {
    die("Valid email is required");
}

if (strlen($_POST["password"]) < 8) {
    die("Password must be at least 8 characters");
}

if ( ! preg_match("/[a-z]/i", $_POST["password"])) {
    die("Password must contain at least one letter");
}

if ( ! preg_match("/[0-9]/", $_POST["password"])) {
    die("Password must contain at least one number");
}

if ($_POST["password"] !== $_POST["cpass"]) {
    die("Passwords must match");
}

$password_hash = password_hash($_POST["password"], PASSWORD_DEFAULT);

require __DIR__ . "/Mysqlconnection.php";

$sql = "INSERT INTO account_info (first_name, last_name,age, email, acc_password)
        VALUES (?, ?, ?, ?, ?)";
        
$stmt = $conn->stmt_init();

if ( ! $stmt->prepare($sql)) {
    die("SQL error: " . $conn->error);
}

$stmt->bind_param("ssiss",
                  $_POST["fn"],
                  $_POST["ln"],
                  $_POST["age"],
                  $_POST["email"],
                  $_POST["password"]);
                  
if ($stmt->execute()) {

    header("Location: signup-success.html");
    exit;
    
} else {
    
    if ($conn->errno === 1062) {
        die("email already taken");
    } else {
        die($conn->error . " " . $conn->errno);
    }
}







