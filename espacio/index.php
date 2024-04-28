<?php

session_start();

if (isset($_SESSION["acc_id"])) {
    
    require __DIR__ . "/Mysqlconnection.php";
    
    $sql = "SELECT * FROM account_info
            WHERE acc_id = {$_SESSION["acc_id"]}";
            
    $result = $conn->query($sql);
    
    $user = $result->fetch_assoc();
}

?>
<!DOCTYPE html>
<html>
<head>
    <title>Home</title>
    <meta charset="UTF-8">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/water.css@2/out/water.css">
</head>
<body>
    
    <h1>Home</h1>
    
    <?php if (isset($user)): ?>
        
        <p>Hello <?= htmlspecialchars($user["first_name"]) ?></p>
        
        <p><a href="logout.php">Log out</a></p>
        
    <?php else: ?>
        
        <p><a href="login.php">Log in</a> or <a href="signup.php">sign up</a></p>
        
    <?php endif; ?>
    
</body>
</html>
    
    
    
    
    
    
    
    
    
    
    