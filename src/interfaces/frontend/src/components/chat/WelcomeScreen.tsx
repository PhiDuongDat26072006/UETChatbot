/**
 * src/interfaces/frontend/src/components/chat/WelcomeScreen.tsx
 * Màn hình chào mừng khi cuộc trò chuyện chưa có tin nhắn, giữ nguyên logo UET,
 * lời chào và 6 thẻ gợi ý câu hỏi học vụ từ index.html gốc.
 */

import { motion } from "motion/react";

export interface WelcomeScreenProps {
  onSelectSuggestion: (query: string) => void;
}

interface SuggestionCard {
  title: string;
  subtitle: string;
  query: string;
}

const SUGGESTION_CARDS: readonly SuggestionCard[] = [
  {
    title: "1. Đăng ký học phần",
    subtitle: "Quy định rút bớt, đăng ký muộn & cải thiện điểm",
    query:
      "Quy định đăng ký học phần, rút bớt học phần và học cải thiện điểm thế nào?",
  },
  {
    title: "2. Chương trình đào tạo",
    subtitle: "Điều kiện làm khóa luận & tiến trình đào tạo tích cực",
    query:
      "Điều kiện để sinh viên K67, K68 ngành Trí tuệ nhân tạo được làm khóa luận tốt nghiệp bảo vệ sớm?",
  },
  {
    title: "3. Lịch học và lịch thi",
    subtitle: "Thủ tục hoãn thi, vắng thi & lịch thi bù học kỳ",
    query:
      "Sinh viên vắng thi có lý do chính đáng hoặc hoãn thi cần làm những thủ tục gì?",
  },
  {
    title: "4. Quy chế sinh viên",
    subtitle: "Cảnh báo học tập, buộc thôi học & xếp loại tốt nghiệp",
    query:
      "Khi nào sinh viên bị cảnh báo học tập và trường hợp nào bị hạ bậc tốt nghiệp?",
  },
  {
    title: "5. Học phí và học bổng",
    subtitle: "Thang điểm GPA, điểm rèn luyện & miễn giảm học phí",
    query:
      "Điều kiện để nhận học bổng khuyến khích học tập loại Giỏi và Xuất sắc tại UET là gì?",
  },
  {
    title: "6. Thông tin phòng ban",
    subtitle: "Địa chỉ liên hệ, phòng ban & mẫu đơn hành chính UET",
    query:
      "Phòng Đào tạo (105-E3) và Phòng Công tác sinh viên (107-G2) tiếp nhận những thủ tục nào?",
  },
];

export function WelcomeScreen({ onSelectSuggestion }: WelcomeScreenProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ type: "spring", stiffness: 360, damping: 28 }}
      style={{ willChange: "transform, opacity" }}
      className="telegram-welcome-screen mx-auto flex w-full max-w-[900px] flex-col items-center px-3 py-5 text-center sm:px-4 sm:py-8"
    >
      <div className="welcome-emblem-wrap mb-3 sm:mb-4">
        <div className="welcome-emblem flex h-[66px] w-[66px] items-center justify-center overflow-hidden rounded-full border-2 border-[rgba(8,117,201,0.2)] bg-white shadow-[0_8px_24px_rgba(0,84,166,0.22)] sm:h-[82px] sm:w-[82px]">
          <img
            src="/static/uet-logo.png?v=v14_robot_mascot"
            alt="UET ChatBot Logo"
            className="uet-logo-img"
          />
        </div>
      </div>

      <h2 className="welcome-headline mb-1.5 text-[19px] font-extrabold tracking-tight text-[var(--text-primary)] sm:mb-2 sm:text-[23px]">
        Xin chào, mình là UET AI Assistant
      </h2>

      <p className="welcome-desc mb-5 max-w-[580px] text-[13px] leading-relaxed text-[var(--text-secondary)] sm:mb-7 sm:text-[14px]">
        Trợ lý thông minh hỗ trợ sinh viên Trường Đại học Công nghệ – ĐHQGHN.
      </p>

      <div className="welcome-suggestions-grid grid w-full max-w-[900px] grid-cols-1 gap-2.5 sm:grid-cols-2 sm:gap-3">
        {SUGGESTION_CARDS.map((card, index) => (
          <motion.button
            key={card.title}
            type="button"
            onClick={() => onSelectSuggestion(card.query)}
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{
              type: "spring",
              stiffness: 400,
              damping: 28,
              delay: index * 0.03,
            }}
            whileHover={{ y: -2 }}
            whileTap={{ scale: 0.985 }}
            style={{ willChange: "transform, opacity" }}
            className="telegram-welcome-card group flex flex-col items-start rounded-xl border border-[var(--border-subtle)] bg-[var(--bg-sidebar)]/90 px-4 py-3.5 text-left shadow-[0_1px_3px_rgba(15,23,42,0.05)] backdrop-blur-sm transition-colors hover:border-[var(--primary)] hover:bg-[var(--uet-blue-tint)] hover:shadow-[0_6px_16px_-2px_rgba(8,117,201,0.14)]"
          >
            <div className="card-text-col min-w-0 w-full">
              <div className="card-title mb-0.5 text-[14px] font-bold text-[var(--text-primary)] group-hover:text-[var(--primary)] dark:group-hover:text-[#38BDF8]">
                {card.title}
              </div>
              <div className="card-subtitle text-[12px] leading-snug text-[var(--text-secondary)]">
                {card.subtitle}
              </div>
            </div>
          </motion.button>
        ))}
      </div>
    </motion.div>
  );
}

