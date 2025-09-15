name := "lab7"
version := "0.1.0"
scalaVersion := "2.12.15"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-sql"   % "3.5.1",
  "org.apache.spark" %% "spark-mllib" % "3.5.1",
  "mysql"            %  "mysql-connector-java" % "8.0.33",
  "com.typesafe"     %  "config" % "1.4.3"
)

Compile / run / fork := true

Compile / run / javaOptions ++= Seq(
  "--add-exports=java.base/sun.nio.ch=ALL-UNNAMED",
  "--add-opens=java.base/java.nio=ALL-UNNAMED",
  "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED"
)

ThisBuild / turbo := true

