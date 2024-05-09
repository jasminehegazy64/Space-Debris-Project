<!DOCTYPE HTML>
<html>

<head>
    <title>ESPACIO</title>
    <meta charset="utf-8">
    <meta name="robots" content="index, follow, max-image-preview:large, max-snippet:-1, max-video-preview:-1">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link rel="stylesheet" href="assets/css/main.css">
</head>

<body class="subpage">

    <!-- Header -->
    <header id="header">
        <div class="inner">
			<a href="index.html" class="logo"><img src="images/logo.png" alt="logo" width="45" height="43"></a>
            <!-- <nav id="nav"><a href="index.html">Home</a> -->
                <!-- <a href="document.html">Documentation</a> -->
                <!-- <a href="contactus.html">Contact Us</a> -->
                <!-- <a href="aboutus.html">About Us</a> -->
                <!-- <a href="reports.html">Reports</a> -->
                <!-- <a href="account.html">Account</a> -->
                <a href="login.php">Sign In</a>
                <a href="signup.php">Sign Up</a>
                <!-- <a href="project.html">Project</a> -->
                <!-- <a href="admindashboard.html">Admin Dashboard</a> -->


            </nav><a href="#navPanel" class="navPanelToggle"><span class="fa fa-bars"></span></a>
        </div>
    </header><!-- Banner -->
    <section id="banner">
        <h1>Welcome to ESPACIO! <br>
            <h3>Please Sign Up to enjoy the amazing features!<h3></h3>
        </h1>
    </section>
    <section id="main" class="wrapper">
        <div class="inner">
            <!-- Form -->
            <form method="post" action="signupconnection.php">
                <center>
                    <!-- <div class="row uniform"> -->
                    <div class="6u$ 12u$(xsmall)">
                        <input type="text" name="fn" id="fname" value="" placeholder="First Name">
                        <?php if (!empty($error_fn)) { echo '<span style="color: red;">' . $error_fn . '</span>'; } ?>
                    </div>
                    <br>
                    <div class="6u 12u$(xsmall)">
                        <input type="text" name="ln" id="Lname" value="" placeholder="Last Name">
                    </div>
                    <br>
                    <div class="6u$ 12u$(xsmall)">
                        <input type="text" name="age" id="age" value="" placeholder="Age">
                    </div>
                    <br>
                    <div class="6u$ 12u$(xsmall)">
                        <input type="email" name="email" id="email" value="" placeholder="Email">
                    </div>
                    <br>
                    <div class="6u 12u$(xsmall)">
                        <input type="password" name="password" id="name" value="" placeholder="Password">
                    </div>
                    <br>
                    <div class="6u$ 12u$(xsmall)">
                        <input type="password" name="cpass" id="cpass" value="" placeholder="Confirm Password">
                    </div>
                    <br>
                    <!-- <div class="6u 12u$(xsmall)">
                        <input type="text" name="affiliation" id="affil" value="" placeholder="Affiliation">
                    </div>
                    <br> -->
                    <!-- <div class="6u 12u$(xsmall)">
                        <input type="text" name="country" id="country" value="" placeholder="Country">
                    </div>
                    <br> -->
                    <div class="12u$">
                        <ul class="actions">
							<li><input type="submit" value="Sign Up"></li>
                            <br><br>
                            Already have an account?
                            <br>
                            <li><a href="login.php"><input type="button" value="Sign In" class="alt"></a></li>

                        </ul>
                    </div>
                </center>
            </form>
        </div>
    </section>
<!-- Footer -->
<footer id="footer">
    <div class="inner">
        <div class="flex">
            <ul class="icons">
                <li><a href="#" class="icon fa-facebook"><span class="label">Facebook</span></a></li>
                <li><a href="#" class="icon fa-twitter"><span class="label">Twitter</span></a></li>
                <li><a href="#" class="icon fa-linkedin"><span class="label">linkedIn</span></a></li>
                <li><a href="#" class="icon fa-pinterest-p"><span class="label">Pinterest</span></a></li>
                <li><a href="#" class="icon fa-vimeo"><span class="label">Vimeo</span></a></li>
            </ul>
        </div>
    </div>
</footer>
<div class="copyright">
    ESPACIO : Detecting, Tracking, and Collision Prediction of Space Debris
</div>

<!-- Scripts -->
<script src="assets/js/jquery.min.js"></script>
<script src="assets/js/skel.min.js"></script>
<script src="assets/js/util.js"></script>
<script src="assets/js/main.js"></script>
</body>

</html>