# Java 时间日期 API


## 📅 Java 时间日期 API


java.time 包 (Java 8+): LocalDate/LocalTime/LocalDateTime、Instant/Duration/Period、DateTimeFormatter、时区 ZonedDateTime、与 Date 互转。


## LocalDate / LocalTime / LocalDateTime


```
// ========== java.time (Java 8+) ==========
// 不可变、线程安全、无 null 问题
// 替代旧的 java.util.Date 和 Calendar

import java.time.*;
import java.time.format.*;
import java.time.temporal.*;

public class DateTimeBasics {
    public static void main(String[] args) {
        // ========== LocalDate (日期) ==========
        LocalDate today = LocalDate.now();           // 2026-04-29
        LocalDate specific = LocalDate.of(2026, 12, 25);  // 2026-12-25
        LocalDate parsed = LocalDate.parse("2026-04-29"); // 字符串解析

        System.out.println(today);         // 2026-04-29
        System.out.println(today.getYear());       // 2026
        System.out.println(today.getMonth());      // APRIL
        System.out.println(today.getMonthValue()); // 4
        System.out.println(today.getDayOfMonth()); // 29
        System.out.println(today.getDayOfWeek());  // WEDNESDAY
        System.out.println(today.lengthOfMonth()); // 30
        System.out.println(today.isLeapYear());    // false

        // 日期加减
        LocalDate tomorrow = today.plusDays(1);
        LocalDate nextWeek = today.plusWeeks(1);
        LocalDate lastMonth = today.minusMonths(1);
        LocalDate nextYear = today.plusYears(1);

        // 日期调整
        LocalDate firstOfMonth = today.withDayOfMonth(1);
        LocalDate lastDayOfMonth = today.with(TemporalAdjusters.lastDayOfMonth());
        LocalDate nextMonday = today.with(TemporalAdjusters.next(DayOfWeek.MONDAY));

        // ========== LocalTime (时间) ==========
        LocalTime now = LocalTime.now();              // 14:30:00.123
        LocalTime time = LocalTime.of(14, 30, 0);     // 14:30
        LocalTime parsedTime = LocalTime.parse("14:30:00");

        System.out.println(now.getHour());     // 14
        System.out.println(now.getMinute());   // 30
        System.out.println(now.getSecond());   // 0

        LocalTime later = now.plusHours(2).plusMinutes(15);

        // ========== LocalDateTime (日期+时间) ==========
        LocalDateTime dt = LocalDateTime.now();
        LocalDateTime dt2 = LocalDateTime.of(2026, 4, 29, 14, 30);
        LocalDateTime dt3 = LocalDateTime.parse("2026-04-29T14:30:00");

        // 分解
        LocalDate datePart = dt.toLocalDate();
        LocalTime timePart = dt.toLocalTime();

        // 组合
        LocalDateTime combined = LocalDateTime.of(today, now);

        // 比较
        System.out.println(dt2.isBefore(dt));   // true
        System.out.println(dt2.isAfter(dt));    // false
        System.out.println(dt2.equals(dt3));    // true
    }
}
```


## Instant & Duration & Period


```
// ========== Instant (时间戳) ==========
// 机器时间: 1970-01-01T00:00:00Z 以来的秒+纳秒

public class InstantDemo {
    public static void main(String[] args) throws Exception {
        Instant now = Instant.now();             // 当前 UTC 时间
        System.out.println(now);                 // 2026-04-29T06:30:00Z

        // 从 Epoch 创建
        Instant epoch = Instant.ofEpochSecond(0);  // 1970-01-01T00:00:00Z
        Instant plus100 = Instant.ofEpochSecond(100_000_000);
        Instant epochMilli = Instant.ofEpochMilli(1700000000000L);

        // 比较
        System.out.println(now.isAfter(epoch));   // true
        System.out.println(now.toEpochMilli());   // 毫秒时间戳

        // ========== Duration (秒纳秒精度的时间差) ==========
        // 用于时间 (LocalTime / Instant)
        Duration d1 = Duration.between(
            LocalTime.of(10, 0), LocalTime.of(14, 30)
        );  // 4小时30分

        Duration d2 = Duration.ofHours(2).plusMinutes(30);
        Duration d3 = Duration.parse("PT4H30M");  // ISO-8601

        System.out.println(d1.toHours());       // 4
        System.out.println(d1.toMinutes());     // 270
        System.out.println(d1.getSeconds());    // 16200
        System.out.println(d1.toMillis());      // 16200000

        // 运算
        Instant start = Instant.now();
        Thread.sleep(100);  // 模拟操作
        Instant end = Instant.now();
        Duration elapsed = Duration.between(start, end);
        System.out.println("Elapsed: " + elapsed.toMillis() + "ms");

        // ========== Period (年月日精度的日期差) ==========
        // 用于日期 (LocalDate)
        Period p1 = Period.between(
            LocalDate.of(2020, 1, 1),
            LocalDate.of(2026, 4, 29)
        );

        Period p2 = Period.ofYears(2).plusMonths(3).plusDays(10);

        System.out.println(p1.getYears());    // 6
        System.out.println(p1.getMonths());   // 3
        System.out.println(p1.getDays());     // 28
        System.out.println(p1);               // P6Y3M28D

        // 计算年龄
        LocalDate birth = LocalDate.of(1990, 6, 15);
        Period age = Period.between(birth, LocalDate.now());
        System.out.printf("Age: %d years %d months%n",
            age.getYears(), age.getMonths());
    }
}
```


## DateTimeFormatter


```
// ========== DateTimeFormatter ==========
// 格式化/解析日期时间

import java.time.format.*;

public class FormatDemo {
    public static void main(String[] args) {
        LocalDateTime now = LocalDateTime.now();

        // ========== 预定义格式 ==========
        System.out.println(DateTimeFormatter.ISO_DATE.format(now));       // 2026-04-29
        System.out.println(DateTimeFormatter.ISO_DATE_TIME.format(now));  // 2026-04-29T14:30:00
        System.out.println(DateTimeFormatter.ISO_LOCAL_DATE.format(now)); // 2026-04-29

        // ========== 自定义格式 ==========
        DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");
        System.out.println(formatter.format(now));  // "2026-04-29 14:30:00"

        // 中文格式
        DateTimeFormatter cnFormat = DateTimeFormatter.ofPattern("yyyy年M月d日 EEEE HH:mm");
        System.out.println(cnFormat.format(now));

        // 常见模式
        // yyyy: 4位年    MM: 2位月    dd: 2位日
        // HH: 24时      hh: 12时    mm: 分      ss: 秒
        // EEEE: 星期全名 EEE: 星期简称
        // a: 上午/下午

        // ========== 解析 ==========
        String dateStr = "2026-04-29";
        LocalDate date = LocalDate.parse(dateStr);  // 默认: yyyy-MM-dd

        String customStr = "2026/04/29 14:30";
        DateTimeFormatter parser = DateTimeFormatter.ofPattern("yyyy/MM/dd HH:mm");
        LocalDateTime parsed = LocalDateTime.parse(customStr, parser);

        // ========== 格式与 Locale ==========
        DateTimeFormatter zhFormatter = DateTimeFormatter
            .ofPattern("yyyy-MM-dd EEEE")
            .withLocale(Locale.CHINESE);
        System.out.println(zhFormatter.format(LocalDate.now()));  // 2026-04-29 星期三
    }
}

// ========== 时区 ==========
class ZoneDemo {
    public static void main(String[] args) {
        // ========== ZonedDateTime ==========
        // LocalDateTime + 时区
        ZonedDateTime nowInBeijing = ZonedDateTime.now(ZoneId.of("Asia/Shanghai"));
        ZonedDateTime nowInUTC = ZonedDateTime.now(ZoneOffset.UTC);
        ZonedDateTime nowInNY = ZonedDateTime.now(ZoneId.of("America/New_York"));

        System.out.println("北京: " + nowInBeijing);
        System.out.println("UTC: " + nowInUTC);
        System.out.println("纽约: " + nowInNY);

        // 时区转换
        ZonedDateTime shanghai = ZonedDateTime.now(ZoneId.of("Asia/Shanghai"));
        ZonedDateTime london = shanghai.withZoneSameInstant(ZoneId.of("Europe/London"));

        // ========== ZoneId ==========
        Set allZones = ZoneId.getAvailableZoneIds();  // 所有可用时区
        System.out.println(allZones.size());  // ~600

        // ========== OffsetDateTime ==========
        // LocalDateTime + 偏移量 (如 +08:00)
        OffsetDateTime offsetDT = OffsetDateTime.now(ZoneOffset.ofHours(8));

        // ========== Date 与 java.time 互转 ==========
        // Date → Instant
        Date oldDate = new Date();
        Instant instant = oldDate.toInstant();

        // Instant → LocalDateTime
        LocalDateTime ldt = LocalDateTime.ofInstant(instant, ZoneId.systemDefault());

        // LocalDateTime → Date
        Date newDate = Date.from(ldt.atZone(ZoneId.systemDefault()).toInstant());
    }
}
```


## 时间 API 总结


```
// ========== java.time 核心类总结 ==========

import java.time.*;
import java.time.format.DateTimeFormatter;
import java.time.temporal.ChronoUnit;

public class DateTimeSummary {
    public static void main(String[] args) {
        // ========== 常用场景 ==========

        // 1. 当前日期时间
        LocalDate.now();
        LocalTime.now();
        LocalDateTime.now();

        // 2. 特定日期
        LocalDate.of(2026, Month.APRIL, 29);
        LocalDate.parse("2026-04-29");

        // 3. 日期加减
        today.plusDays(1).minusMonths(2).plusYears(1);

        // 4. 时间差
        Duration.between(start, end);
        Period.between(birth, today);

        // 5. 时间戳
        Instant.now().toEpochMilli();

        // 6. 格式化
        DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss").format(dt);

        // 7. 时区转换
        ZonedDateTime.now(ZoneId.of("Asia/Shanghai"));

        // 8. 两个日期相隔天数
        long daysBetween = ChronoUnit.DAYS.between(
            LocalDate.of(2026, 1, 1), LocalDate.now()
        );

        // ========== 旧 API (不要用!) ==========
        // ❌ java.util.Date — 设计差, 可变, 不线程安全
        // ❌ java.util.Calendar — 复杂, 月份从0开始
        // ❌ SimpleDateFormat — 不线程安全
        // ✅ 全部用 java.time 替代

        // ========== 新旧对照 ==========
        // Date               → Instant
        // Calendar           → LocalDate / ZonedDateTime
        // SimpleDateFormat   → DateTimeFormatter
        // Timestamp          → Instant + 时区

        // ========== 选择指南 ==========
        // 日期+时间+时区 → ZonedDateTime
        // 日期+时间      → LocalDateTime
        // 只有日期       → LocalDate
        // 只有时间       → LocalTime
        // 时间戳         → Instant
        // 时间差         → Duration (秒纳秒) / Period (年月日)
        // 格式化/解析    → DateTimeFormatter
    }
}
```


> **Note:** 💡 时间 API 要点: java.time (Java 8+) 替代 Date/Calendar; LocalDate/LocalTime/LocalDateTime 不可变; Instant 时间戳; Duration 秒级差, Period 日期差; DateTimeFormatter 线程安全; ZonedDateTime 时区; 与旧 Date 互转通过 Instant; 格式化.ofPattern("yyyy-MM-dd"); 计算用 ChronoUnit 或 Duration/Period。


## 练习


<!-- Converted from: 34_Java 时间日期 API.html -->
