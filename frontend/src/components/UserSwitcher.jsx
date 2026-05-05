/**
 * UserSwitcher.jsx — Top-right dropdown to pick the active user.
 *
 * In dev mode, lists Alice/Bob/Test (the dev users).
 * In Phase 3, this becomes a real "logged in as ..." display
 * with a logout option, using MS Graph identity.
 */
import React from 'react';
import {
  Menu,
  MenuTrigger,
  MenuPopover,
  MenuList,
  MenuItem,
  Button,
  Avatar,
  Text,
  tokens,
  makeStyles,
  Spinner,
} from '@fluentui/react-components';
import {
  Person24Regular,
  ChevronDown16Regular,
  Checkmark16Regular,
} from '@fluentui/react-icons';

import { useUser } from '../context/UserContext';


const useStyles = makeStyles({
  trigger: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    paddingLeft: '8px',
    paddingRight: '8px',
  },
  triggerName: {
    fontWeight: 500,
  },
  menuItemRow: {
    display: 'flex',
    alignItems: 'center',
    gap: '10px',
    width: '100%',
  },
  menuItemInfo: {
    display: 'flex',
    flexDirection: 'column',
    gap: '2px',
    flex: 1,
  },
  menuItemEmail: {
    color: tokens.colorNeutralForeground3,
    fontSize: '12px',
  },
  errorBox: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    color: tokens.colorPaletteRedForeground1,
    padding: '0 12px',
  },
});


// Generate a consistent color for each user (used in Avatar)
const AVATAR_COLORS = ['brand', 'colorful', 'beige', 'cornflower', 'lavender'];
function colorFor(userId) {
  if (!userId) return 'neutral';
  // Hash userId to pick a color
  let hash = 0;
  for (let i = 0; i < userId.length; i++) {
    hash = (hash << 5) - hash + userId.charCodeAt(i);
    hash |= 0;
  }
  return AVATAR_COLORS[Math.abs(hash) % AVATAR_COLORS.length];
}


export default function UserSwitcher() {
  const styles = useStyles();
  const { activeUser, activeUserId, setActiveUserId, users, loading, error } = useUser();

  // ── Loading state ────────────────────────────────────────────
  if (loading) {
    return (
      <div className={styles.trigger}>
        <Spinner size="tiny" />
        <Text>Loading...</Text>
      </div>
    );
  }

  // ── Error state ──────────────────────────────────────────────
  if (error) {
    return (
      <div className={styles.errorBox}>
        <Person24Regular />
        <Text size={200}>API offline</Text>
      </div>
    );
  }

  // ── Empty state ──────────────────────────────────────────────
  if (!users || users.length === 0) {
    return (
      <div className={styles.trigger}>
        <Person24Regular />
        <Text>No users</Text>
      </div>
    );
  }

  // ── Dropdown ─────────────────────────────────────────────────
  const displayName = activeUser?.display_name || activeUserId || 'User';

  return (
    <Menu>
      <MenuTrigger disableButtonEnhancement>
        <Button
          appearance="subtle"
          icon={
            <Avatar
              size={28}
              name={displayName}
              color={colorFor(activeUserId)}
            />
          }
          iconPosition="before"
        >
          <span className={styles.triggerName}>{displayName}</span>
          <ChevronDown16Regular style={{ marginLeft: 6 }} />
        </Button>
      </MenuTrigger>

      <MenuPopover>
        <MenuList>
          {users.map((u) => {
            const isActive = u.id === activeUserId;
            const name = u.display_name || u.id;
            return (
              <MenuItem
                key={u.id}
                onClick={() => setActiveUserId(u.id)}
                icon={
                  <Avatar
                    size={32}
                    name={name}
                    color={colorFor(u.id)}
                  />
                }
                secondaryContent={
                  isActive ? <Checkmark16Regular /> : null
                }
              >
                <div className={styles.menuItemInfo}>
                  <Text weight={isActive ? 'semibold' : 'regular'}>
                    {name}
                  </Text>
                  {u.email && (
                    <Text className={styles.menuItemEmail}>{u.email}</Text>
                  )}
                </div>
              </MenuItem>
            );
          })}
        </MenuList>
      </MenuPopover>
    </Menu>
  );
}