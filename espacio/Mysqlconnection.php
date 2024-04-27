<?php
// MySQL server configuration
$servername = "localhost";
$username = "root";
$password = "Happylola.123";
$database = "espacio";

// Create connection
$conn = mysqli_connect($servername, $username, $password, $database);

// Check connection
if (!$conn) {
    die("Connection failed: " . mysqli_connect_error());
} else {
    echo "Connected successfully";
}

?>
