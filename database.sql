CREATE DATABASE espacio;
USE espacio;
CREATE TABLE account_info (
  acc_id int NOT NULL AUTO_INCREMENT,
  first_name varchar(50) DEFAULT NULL,
  last_name varchar(50) DEFAULT NULL,
  age int DEFAULT NULL,
  username varchar(10) DEFAULT NULL,
  email varchar(100) NOT NULL,
  acc_password varchar(50) NOT NULL,
  confirm_pass varchar(10) DEFAULT NULL,
  PRIMARY KEY (acc_id)
) ENGINE=InnoDB AUTO_INCREMENT=2 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci